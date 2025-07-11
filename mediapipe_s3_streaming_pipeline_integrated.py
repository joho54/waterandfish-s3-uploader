import os
import sys
import json
import argparse
import logging
import pickle
import gzip
import io
import boto3
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from config import *
from scipy.interpolate import interp1d
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import holistic as mp_holistic



class MediaPipeManager:
    """MediaPipe 객체를 안전하게 관리하는 컨텍스트 매니저"""

    _instance = None
    _holistic = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MediaPipeManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._holistic is None:
            self._holistic = mp_holistic.Holistic(
                static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
                model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
                smooth_landmarks=MEDIAPIPE_SMOOTH_LANDMARKS,
                enable_segmentation=MEDIAPIPE_ENABLE_SEGMENTATION,
                smooth_segmentation=MEDIAPIPE_SMOOTH_SEGMENTATION,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )

    def __enter__(self):
        return self._holistic

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 전역 객체는 유지하고 정리만
        pass

    @classmethod
    def cleanup(cls):
        """전역 MediaPipe 객체 정리"""
        if cls._holistic:
            cls._holistic.close()
            cls._holistic = None

# S3 업로더 클래스
class S3StreamingUploader:
    def __init__(self, bucket_name, prefix, region_name=None):
        self.s3 = boto3.client('s3', region_name=region_name)
        self.bucket = bucket_name
        self.prefix = prefix

    def upload_pickle_gzip(self, obj, s3_key):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buf.read())


def interpolate_individual_landmarks(landmarks_list):
    """개별 랜드마크 포인트의 결측치를 interpolation으로 보완합니다."""
    if not landmarks_list or len(landmarks_list) < 2:
        return landmarks_list
    
    # 각 랜드마크 타입별 포인트 수
    landmark_counts = {"pose": 33, "left_hand": 21, "right_hand": 21}
    
    # 각 타입별로 interpolation 수행
    for landmark_type in ["pose", "left_hand", "right_hand"]:
        num_points = landmark_counts[landmark_type]
        
        # 각 포인트별로 시간축 interpolation
        for point_idx in range(num_points):
            # 해당 포인트의 모든 프레임에서의 좌표 수집
            x_coords, y_coords, z_coords = [], [], []
            valid_frames = []
            
            for frame_idx, frame in enumerate(landmarks_list):
                if frame.get(landmark_type):
                    if isinstance(frame[landmark_type], list):
                        if point_idx < len(frame[landmark_type]):
                            point = frame[landmark_type][point_idx]
                            x_coords.append(point[0])
                            y_coords.append(point[1])
                            z_coords.append(point[2])
                            valid_frames.append(frame_idx)
                    else:
                        # MediaPipe landmark 객체인 경우
                        landmarks = frame[landmark_type].landmark
                        if point_idx < len(landmarks):
                            x_coords.append(landmarks[point_idx].x)
                            y_coords.append(landmarks[point_idx].y)
                            z_coords.append(landmarks[point_idx].z)
                            valid_frames.append(frame_idx)
            
            # 유효한 프레임이 2개 이상일 때만 interpolation 수행
            if len(valid_frames) >= 2:
                # 시간축 interpolation
                x_interp = interp1d(valid_frames, x_coords, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                y_interp = interp1d(valid_frames, y_coords, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                z_interp = interp1d(valid_frames, z_coords, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                
                # 모든 프레임에 대해 보간된 값 적용
                for frame_idx in range(len(landmarks_list)):
                    if frame_idx not in valid_frames:
                        # 결측 프레임에 보간된 값 적용
                        interpolated_x = float(x_interp(frame_idx))
                        interpolated_y = float(y_interp(frame_idx))
                        interpolated_z = float(z_interp(frame_idx))
                        
                        # 기존 프레임에 해당 타입이 없으면 생성
                        if not landmarks_list[frame_idx].get(landmark_type):
                            landmarks_list[frame_idx][landmark_type] = []
                        
                        # 리스트 형태로 변환
                        if not isinstance(landmarks_list[frame_idx][landmark_type], list):
                            landmarks_list[frame_idx][landmark_type] = [
                                [l.x, l.y, l.z] for l in landmarks_list[frame_idx][landmark_type].landmark
                            ]
                        
                        # 포인트 개수 맞추기
                        while len(landmarks_list[frame_idx][landmark_type]) <= point_idx:
                            landmarks_list[frame_idx][landmark_type].append([0, 0, 0])
                        
                        # 보간된 값 적용
                        landmarks_list[frame_idx][landmark_type][point_idx] = [
                            interpolated_x, interpolated_y, interpolated_z
                        ]
    
    return landmarks_list


def apply_temporal_smoothing(landmarks_list, window_size=3, alpha=0.7):
    """시간적 smoothing을 적용하여 랜드마크 변화를 부드럽게 만듭니다."""
    if not landmarks_list or len(landmarks_list) < 2:
        return landmarks_list
    
    smoothed_landmarks = []
    landmark_counts = {"pose": 33, "left_hand": 21, "right_hand": 21}
    
    for frame_idx, frame in enumerate(landmarks_list):
        smoothed_frame = {}
        
        for landmark_type in ["pose", "left_hand", "right_hand"]:
            if not frame.get(landmark_type):
                smoothed_frame[landmark_type] = None
                continue
            
            num_points = landmark_counts[landmark_type]
            smoothed_landmarks_type = []
            
            # 리스트 형태로 변환
            if not isinstance(frame[landmark_type], list):
                current_landmarks = [[l.x, l.y, l.z] for l in frame[landmark_type].landmark]
            else:
                current_landmarks = frame[landmark_type].copy()
            
            # 각 포인트별로 smoothing 적용
            for point_idx in range(num_points):
                if point_idx >= len(current_landmarks):
                    current_landmarks.append([0, 0, 0])
                
                # 윈도우 내의 이전 프레임들 수집
                window_coords = []
                for w in range(max(0, frame_idx - window_size + 1), frame_idx + 1):
                    if w < len(landmarks_list) and landmarks_list[w].get(landmark_type):
                        if isinstance(landmarks_list[w][landmark_type], list):
                            if point_idx < len(landmarks_list[w][landmark_type]):
                                window_coords.append(landmarks_list[w][landmark_type][point_idx])
                        else:
                            landmarks = landmarks_list[w][landmark_type].landmark
                            if point_idx < len(landmarks):
                                window_coords.append([landmarks[point_idx].x, 
                                                    landmarks[point_idx].y, 
                                                    landmarks[point_idx].z])
                
                if window_coords:
                    # 가중 평균 계산 (최근 프레임에 더 높은 가중치)
                    weights = np.exp(alpha * np.arange(len(window_coords)))
                    weights = weights / np.sum(weights)
                    
                    smoothed_point = [0, 0, 0]
                    for i, coord in enumerate(window_coords):
                        for j in range(3):
                            smoothed_point[j] += coord[j] * weights[i]
                    
                    smoothed_landmarks_type.append(smoothed_point)
                else:
                    smoothed_landmarks_type.append(current_landmarks[point_idx])
            
            smoothed_frame[landmark_type] = smoothed_landmarks_type
        
        smoothed_landmarks.append(smoothed_frame)
    
    return smoothed_landmarks


def check_spatial_consistency_and_correct(landmarks_list):
    """공간적 일관성을 검사하고 보정합니다."""
    if not landmarks_list:
        return landmarks_list
    
    corrected_landmarks = []
    landmark_counts = {"pose": 33, "left_hand": 21, "right_hand": 21}
    
    # 손목-손가락 연결성 검사 및 보정
    hand_connections = {
        "left_hand": [(0, 1), (1, 2), (2, 3), (3, 4),  # 엄지
                     (0, 5), (5, 6), (6, 7), (7, 8),  # 검지
                     (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
                     (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
                     (0, 17), (17, 18), (18, 19), (19, 20)],  # 새끼
        "right_hand": [(0, 1), (1, 2), (2, 3), (3, 4),  # 엄지
                      (0, 5), (5, 6), (6, 7), (7, 8),  # 검지
                      (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
                      (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
                      (0, 17), (17, 18), (18, 19), (19, 20)]  # 새끼
    }
    
    for frame_idx, frame in enumerate(landmarks_list):
        corrected_frame = {}
        
        for landmark_type in ["pose", "left_hand", "right_hand"]:
            if not frame.get(landmark_type):
                corrected_frame[landmark_type] = None
                continue
            
            # 리스트 형태로 변환
            if not isinstance(frame[landmark_type], list):
                current_landmarks = [[l.x, l.y, l.z] for l in frame[landmark_type].landmark]
            else:
                current_landmarks = frame[landmark_type].copy()
            
            # 손의 경우 연결성 검사 및 보정
            if landmark_type in ["left_hand", "right_hand"]:
                corrected_landmarks_type = current_landmarks.copy()
                
                # 각 연결에 대해 거리 검사
                for start_idx, end_idx in hand_connections[landmark_type]:
                    if (start_idx < len(corrected_landmarks_type) and 
                        end_idx < len(corrected_landmarks_type)):
                        
                        start_point = corrected_landmarks_type[start_idx]
                        end_point = corrected_landmarks_type[end_idx]
                        
                        # 거리 계산
                        distance = np.sqrt(sum((np.array(start_point) - np.array(end_point))**2))
                        
                        # 비정상적으로 긴 거리인 경우 보정
                        max_reasonable_distance = 0.3  # 임계값
                        if distance > max_reasonable_distance:
                            # 중간점으로 보정
                            mid_point = [(start_point[i] + end_point[i]) / 2 for i in range(3)]
                            
                            # 시작점과 끝점을 중간점으로부터 적절한 거리로 조정
                            direction = np.array(end_point) - np.array(start_point)
                            if np.linalg.norm(direction) > 0:
                                direction = direction / np.linalg.norm(direction)
                                corrected_distance = max_reasonable_distance / 2
                                
                                corrected_landmarks_type[start_idx] = [
                                    mid_point[i] - direction[i] * corrected_distance for i in range(3)
                                ]
                                corrected_landmarks_type[end_idx] = [
                                    mid_point[i] + direction[i] * corrected_distance for i in range(3)
                                ]
                
                corrected_frame[landmark_type] = corrected_landmarks_type
            else:
                # 포즈의 경우 기본 검사
                corrected_frame[landmark_type] = current_landmarks
        
        corrected_landmarks.append(corrected_frame)
    
    return corrected_landmarks


def convert_to_relative_coordinates(landmarks_list):
    """절대 좌표를 어깨 중심 상대 좌표계로 변환합니다."""
    relative_landmarks = []

    for frame in landmarks_list:
        if not frame["pose"]:
            relative_landmarks.append(frame)
            continue

        # 랜드마크 데이터가 리스트인지 MediaPipe 객체인지 확인
        if isinstance(frame["pose"], list):
            pose_landmarks = frame["pose"]
            # 리스트 형태의 랜드마크에서 어깨 좌표 추출
            if len(pose_landmarks) > 12:
                left_shoulder = pose_landmarks[11]
                right_shoulder = pose_landmarks[12]
                shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
                shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
                shoulder_center_z = (left_shoulder[2] + right_shoulder[2]) / 2
                shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            else:
                # 어깨 랜드마크가 없는 경우 기본값 사용
                shoulder_center_x, shoulder_center_y, shoulder_center_z = 0, 0, 0
                shoulder_width = 1.0
        else:
            # MediaPipe 객체인 경우
            pose_landmarks = frame["pose"].landmark
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
            shoulder_width = abs(right_shoulder.x - left_shoulder.x)

        if shoulder_width == 0:
            shoulder_width = 1.0

        new_frame = {}

        if frame["pose"]:
            relative_pose = []
            if isinstance(frame["pose"], list):
                for landmark in frame["pose"]:
                    rel_x = (landmark[0] - shoulder_center_x) / shoulder_width
                    rel_y = (landmark[1] - shoulder_center_y) / shoulder_width
                    rel_z = (landmark[2] - shoulder_center_z) / shoulder_width
                    relative_pose.append([rel_x, rel_y, rel_z])
            else:
                for landmark in frame["pose"].landmark:
                    rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                    rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                    rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                    relative_pose.append([rel_x, rel_y, rel_z])
            new_frame["pose"] = relative_pose

        for hand_key in ["left_hand", "right_hand"]:
            if frame[hand_key]:
                relative_hand = []
                if isinstance(frame[hand_key], list):
                    for landmark in frame[hand_key]:
                        rel_x = (landmark[0] - shoulder_center_x) / shoulder_width
                        rel_y = (landmark[1] - shoulder_center_y) / shoulder_width
                        rel_z = (landmark[2] - shoulder_center_z) / shoulder_width
                        relative_hand.append([rel_x, rel_y, rel_z])
                else:
                    for landmark in frame[hand_key].landmark:
                        rel_x = (landmark.x - shoulder_center_x) / shoulder_width
                        rel_y = (landmark.y - shoulder_center_y) / shoulder_width
                        rel_z = (landmark.z - shoulder_center_z) / shoulder_width
                        relative_hand.append([rel_x, rel_y, rel_z])
                new_frame[hand_key] = relative_hand
            else:
                new_frame[hand_key] = None

        relative_landmarks.append(new_frame)

    return relative_landmarks

def enhanced_preprocess_landmarks(landmarks_list):
    """개선된 랜드마크 전처리 함수 - interpolation, smoothing, 일관성 검사 포함."""
    if not landmarks_list:
        return np.zeros((TARGET_SEQ_LENGTH, 675))

    print("    🔧 개별 랜드마크 interpolation 적용 중...")
    # 1. 개별 랜드마크 포인트 interpolation
    interpolated_landmarks = interpolate_individual_landmarks(landmarks_list)
    
    print("    🔧 시간적 smoothing 적용 중...")
    # 2. 시간적 smoothing 적용
    smoothed_landmarks = apply_temporal_smoothing(interpolated_landmarks)
    
    print("    🔧 공간적 일관성 검사 및 보정 중...")
    # 3. 공간적 일관성 검사 및 보정
    corrected_landmarks = check_spatial_consistency_and_correct(smoothed_landmarks)
    
    # 기존 전처리 로직 적용
    relative_landmarks = convert_to_relative_coordinates(corrected_landmarks)

    processed_frames = []
    for frame in relative_landmarks:
        combined = []
        for key in ["pose", "left_hand", "right_hand"]:
            if frame[key]:
                # convert_to_relative_coordinates에서 이미 리스트 형태로 변환됨
                if isinstance(frame[key], list):
                    combined.extend(frame[key])
                else:
                    # 혹시 MediaPipe 객체가 남아있다면 변환
                    combined.extend([[l.x, l.y, l.z] for l in frame[key].landmark])
            else:
                num_points = {"pose": 33, "left_hand": 21, "right_hand": 21}[key]
                combined.extend([[0, 0, 0]] * num_points)

        if combined:
            processed_frames.append(np.array(combined).flatten())
        else:
            processed_frames.append(np.zeros(75 * 3))

    if not processed_frames:
        return np.zeros((TARGET_SEQ_LENGTH, 675))

    sequence = np.array(processed_frames)

    if len(sequence) > 0:
        try:
            sequence = normalize_sequence_length(sequence, TARGET_SEQ_LENGTH)
            
            # 기본 sequence 구조 검증 (속도, 가속도 추가 전)
            print("    🔍 기본 sequence 구조 검증 중...")
            validate_sequence_structure(sequence)
            
            sequence = extract_dynamic_features(sequence)

            # 정규화 개선: 더 강한 정규화
            sequence = (sequence - np.mean(sequence)) / (np.std(sequence) + 1e-8)

            return sequence
        except Exception as e:
            print(f"⚠️ 시퀀스 처리 중 오류 발생: {e}")
            return np.zeros((TARGET_SEQ_LENGTH, 675))

    return np.zeros((TARGET_SEQ_LENGTH, 675))


def normalize_sequence_length(sequence, target_length=30):
    """시퀀스 길이를 정규화합니다."""
    current_length = len(sequence)

    if current_length == target_length:
        return sequence

    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)

    normalized_sequence = []
    for i in range(sequence.shape[1]):
        f = interp1d(
            x_old,
            sequence[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        normalized_sequence.append(f(x_new))

    return np.array(normalized_sequence).T



def extract_dynamic_features(sequence):
    """속도와 가속도 특징을 추출합니다."""
    velocity = np.diff(sequence, axis=0, prepend=sequence[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    dynamic_features = np.concatenate([sequence, velocity, acceleration], axis=1)
    return dynamic_features


def validate_sequence_structure(sequence, frame_idx=0):
    """sequence의 구조가 pose, left_hand, right_hand로 올바르게 구성되었는지 검증합니다."""
    if sequence is None or len(sequence) == 0:
        print("❌ sequence가 비어있습니다.")
        return False
    
    # 기본 sequence (속도, 가속도 제외)
    base_sequence_length = sequence.shape[1] // 3  # velocity, acceleration 제외
    frame_data = sequence[frame_idx][:base_sequence_length]
    
    # 예상되는 랜드마크 개수
    expected_counts = {
        "pose": 33,
        "left_hand": 21, 
        "right_hand": 21
    }
    
    total_expected = sum(expected_counts.values()) * 3  # x, y, z 좌표
    
    print(f"🔍 Sequence 구조 검증:")
    print(f"   📊 전체 sequence 형태: {sequence.shape}")
    print(f"   📊 기본 sequence 길이: {base_sequence_length}")
    print(f"   📊 예상 랜드마크 개수: {total_expected}")
    print(f"   📊 실제 데이터 길이: {len(frame_data)}")
    
    if len(frame_data) != total_expected:
        print(f"❌ 랜드마크 개수 불일치: 예상 {total_expected}, 실제 {len(frame_data)}")
        return False
    
    # 각 랜드마크 타입별로 데이터 확인
    start_idx = 0
    for landmark_type, count in expected_counts.items():
        end_idx = start_idx + count * 3
        landmark_data = frame_data[start_idx:end_idx]
        
        print(f"   📍 {landmark_type}: {count}개 랜드마크, {len(landmark_data)}개 좌표값")
        print(f"      범위: {start_idx} ~ {end_idx-1}")
        print(f"      샘플 데이터: {landmark_data[:6]}...")  # 처음 6개 값 (2개 랜드마크)
        
        start_idx = end_idx
    
    print("✅ Sequence 구조가 올바르게 구성되었습니다.")
    return True



def generate_balanced_none_class_data(file_mapping, none_class, target_count=None):
    """다른 클래스와 균형있는 None 클래스 데이터를 생성하고 캐시에 저장합니다."""
    print(f"\n✨ '{none_class}' 클래스 데이터 생성 중...")

    # 목표 개수 계산 (다른 클래스의 평균 개수)
    if target_count is None:
        # 다른 클래스들의 원본 파일 개수 계산
        other_class_counts = []
        for filename, info in file_mapping.items():
            if info["label"] != none_class:
                other_class_counts.append(info["label"])

        # 라벨별 개수 집계
        from collections import Counter

        label_counts = Counter(other_class_counts)

        if label_counts:
            # 다른 클래스들의 평균 개수 계산 (증강 후 예상 개수)
            avg_original_count = sum(label_counts.values()) / len(label_counts)
            target_count = int(avg_original_count * (1 + AUGMENTATIONS_PER_VIDEO))
            print(
                f"📊 다른 클래스 평균: {avg_original_count:.1f}개 → 목표 None 클래스: {target_count}개"
            )
        else:
            target_count = 100  # 기본값
            print(f"📊 기본 목표 None 클래스: {target_count}개")

    none_samples = []
    source_videos = list(file_mapping.keys())

    # 목표 개수에 도달할 때까지 반복
    video_index = 0
    while len(none_samples) < target_count and video_index < len(source_videos):
        filename = source_videos[video_index % len(source_videos)]  # 순환 사용
        file_path = file_mapping[filename]["path"]

        try:
            # MediaPipe 객체 재사용 (한 번에 하나씩 처리)
            with MediaPipeManager() as holistic:
                landmarks = extract_landmarks_with_holistic(file_path, holistic)

                if landmarks and len(landmarks) > 10:
                    # 영상의 시작, 1/4, 1/2, 3/4, 끝 지점에서 프레임 추출
                    frame_indices = [
                        0,
                        len(landmarks) // 4,
                        len(landmarks) // 2,
                        3 * len(landmarks) // 4,
                        -1,
                    ]

                    for idx in frame_indices:
                        if len(none_samples) >= target_count:
                            break

                        static_landmarks = [landmarks[idx]] * TARGET_SEQ_LENGTH
                        static_sequence = enhanced_preprocess_landmarks(
                            static_landmarks
                        )

                        if static_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                            continue

                        # 정적 시퀀스 추가
                        none_samples.append(static_sequence)

                        # 미세한 움직임 추가 (노이즈) - 목표 개수 제한
                        for _ in range(
                            min(
                                NONE_CLASS_AUGMENTATIONS_PER_FRAME,
                                target_count - len(none_samples),
                            )
                        ):
                            if len(none_samples) >= target_count:
                                break
                            augmented = augment_sequence_improved(
                                static_sequence, noise_level=NONE_CLASS_NOISE_LEVEL
                            )
                            if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                                none_samples.append(augmented)

                    # 느린 전환 데이터 생성 (목표 개수 제한)
                    if len(none_samples) < target_count:
                        start_frame_lm = landmarks[0]
                        middle_frame_lm = landmarks[len(landmarks) // 2]

                        transition_landmarks = []
                        for i in range(TARGET_SEQ_LENGTH):
                            alpha = i / (TARGET_SEQ_LENGTH - 1)
                            interp_frame = {}
                            for key in ["pose", "left_hand", "right_hand"]:
                                if start_frame_lm.get(key) and middle_frame_lm.get(key):
                                    interp_lm = []
                                    
                                    # 랜드마크 데이터가 리스트인지 MediaPipe 객체인지 확인
                                    if isinstance(start_frame_lm[key], list):
                                        start_lms = start_frame_lm[key]
                                        mid_lms = middle_frame_lm[key]
                                        for j in range(len(start_lms)):
                                            new_x = (
                                                start_lms[j][0] * (1 - alpha)
                                                + mid_lms[j][0] * alpha
                                            )
                                            new_y = (
                                                start_lms[j][1] * (1 - alpha)
                                                + mid_lms[j][1] * alpha
                                            )
                                            new_z = (
                                                start_lms[j][2] * (1 - alpha)
                                                + mid_lms[j][2] * alpha
                                            )
                                            interp_lm.append([new_x, new_y, new_z])
                                    else:
                                        # MediaPipe 객체인 경우
                                        start_lms = start_frame_lm[key].landmark
                                        mid_lms = middle_frame_lm[key].landmark
                                        for j in range(len(start_lms)):
                                            new_x = (
                                                start_lms[j].x * (1 - alpha)
                                                + mid_lms[j].x * alpha
                                            )
                                            new_y = (
                                                start_lms[j].y * (1 - alpha)
                                                + mid_lms[j].y * alpha
                                            )
                                            new_z = (
                                                start_lms[j].z * (1 - alpha)
                                                + mid_lms[j].z * alpha
                                            )
                                            interp_lm.append([new_x, new_y, new_z])
                                    
                                    interp_frame[key] = interp_lm
                                else:
                                    interp_frame[key] = None
                            transition_landmarks.append(interp_frame)

                        transition_sequence = enhanced_preprocess_landmarks(
                            transition_landmarks
                        )
                        if transition_sequence.shape == (TARGET_SEQ_LENGTH, 675):
                            none_samples.append(transition_sequence)

        except Exception as e:
            print(f"⚠️ None 클래스 데이터 생성 중 오류: {filename}, 오류: {e}")

        video_index += 1

    print(
        f"✅ {none_class} 클래스 데이터 생성 완료: {len(none_samples)}개 샘플 (목표: {target_count}개)"
    )

    return none_samples

def validate_video_roots():
    """VIDEO_ROOTS의 모든 디렉토리가 존재하는지 확인합니다."""
    print("🔍 비디오 루트 디렉토리 검증 중...")
    valid_roots = []

    for (range_start, range_end), root_path in VIDEO_ROOTS:
        if os.path.exists(root_path):
            valid_roots.append(((range_start, range_end), root_path))
            print(f"✅ {range_start}~{range_end}: {root_path}")
        else:
            print(f"❌ {range_start}~{range_end}: {root_path} (존재하지 않음)")

    return valid_roots


def get_video_root_and_path(filename):
    """파일명에서 번호를 추출해 올바른 VIDEO_ROOT 경로와 실제 파일 경로를 반환합니다."""
    try:
        # 파일 확장자 제거
        file_id = filename.split(".")[0]

        # KETI_SL_ 형식 확인
        if not file_id.startswith("KETI_SL_"):
            print(f"⚠️ KETI_SL_ 형식이 아닌 파일명: {filename}")
            return None

        # 숫자 부분 추출
        number_str = file_id.replace("KETI_SL_", "")
        if not number_str.isdigit():
            print(f"⚠️ 숫자가 아닌 파일명: {filename}")
            return None

        num = int(number_str)

        # 적절한 디렉토리 찾기
        target_root = None
        for (range_start, range_end), root_path in VIDEO_ROOTS:
            if range_start <= num <= range_end:
                target_root = root_path
                break

        if target_root is None:
            print(f"⚠️ 번호 {num}에 해당하는 디렉토리를 찾을 수 없음: {filename}")
            return None

        # 파일 찾기
        file_path = find_file_in_directory(target_root, filename)
        if file_path:
            return file_path

        print(f"⚠️ 파일을 찾을 수 없음: {filename} (디렉토리: {target_root})")
        return None

    except Exception as e:
        print(f"⚠️ 파일명 파싱 오류: {filename}, 오류: {e}")
        return None
    
    
def find_file_in_directory(directory, filename_pattern):
    """디렉토리에서 파일 패턴에 맞는 파일을 찾습니다."""
    if not os.path.exists(directory):
        return None

    # 파일명에서 확장자 제거
    base_name = filename_pattern.split(".")[0]

    # 가능한 확장자들 (config에서 가져옴)
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(directory, base_name + ext)
        if os.path.exists(candidate):
            return candidate

    return None



def extract_landmarks_with_holistic(video_path, holistic):
    """전달받은 MediaPipe 객체를 사용하여 랜드마크를 추출합니다."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ 비디오 파일을 열 수 없음: {video_path}")
            return None

        # 비디오 정보 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"    📊 비디오 정보: {total_frames}프레임, {fps:.1f}fps")

        landmarks_list = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            frame_data = {
                "pose": results.pose_landmarks,
                "left_hand": results.left_hand_landmarks,
                "right_hand": results.right_hand_landmarks,
            }
            landmarks_list.append(frame_data)
            frame_count += 1
            
        cap.release()
        print(f"    ✅ 랜드마크 추출 완료: {len(landmarks_list)}프레임")
        return landmarks_list

    except (cv2.error, OSError) as e:
        print(f"⚠️ 비디오 파일 읽기 오류: {video_path}, 오류: {e}")
        return None
    except Exception as e:
        print(f"⚠️ 랜드마크 추출 중 예상치 못한 오류: {video_path}, 오류: {e}")
        return None

def extract_and_cache_label_data_optimized(file_mapping, label):
    """메모리 효율적인 라벨별 데이터 추출 및 캐싱"""
    print(f"\n🔄 {label} 라벨 데이터 추출 중...")

    # 해당 라벨의 파일들만 필터링
    label_files = {
        filename: info
        for filename, info in file_mapping.items()
        if info["label"] == label
    }

    if not label_files:
        print(f"⚠️ {label} 라벨에 해당하는 파일이 없습니다.")
        return []

    label_data = []

    # 배치 단위로 처리
    for batch in process_data_in_batches(
        label_files, batch_size=BATCH_SIZE_FOR_PROCESSING
    ):
        for item in batch:
            if item["label"] == label:
                # 원본 데이터 추가
                label_data.append(item["sequence"])

                # 증강 데이터 추가
                for _ in range(AUGMENTATIONS_PER_VIDEO):
                    try:
                        augmented = augment_sequence_improved(item["sequence"])
                        if augmented.shape == (TARGET_SEQ_LENGTH, 675):
                            label_data.append(augmented)
                    except Exception as e:
                        print(f"⚠️ 증강 중 오류: {e}")
                        continue

    print(f"✅ {label} 라벨 데이터 추출 완료: {len(label_data)}개 샘플")

    # 캐시에 저장
    save_label_cache(label, label_data)

    return label_data

def process_data_in_batches(file_mapping, batch_size=100):
    """메모리 효율성을 위해 데이터를 배치 단위로 처리합니다."""
    all_files = list(file_mapping.items())
    total_files = len(all_files)

    print(f"📊 총 {total_files}개 파일을 {batch_size}개씩 배치 처리합니다.")

    # 진행률 표시 설정에 따라 tqdm 사용
    if ENABLE_PROGRESS_BAR:
        iterator = tqdm(range(0, total_files, batch_size), desc="배치 처리")
    else:
        iterator = range(0, total_files, batch_size)

    # MediaPipe 객체 재사용
    try:
        with MediaPipeManager() as holistic:
            print("✅ MediaPipe 객체 초기화 완료")

            for i in iterator:
                batch_files = all_files[i : i + batch_size]
                batch_data = []

                print(
                    f"🔄 배치 {i//batch_size + 1} 처리 중... ({len(batch_files)}개 파일)"
                )

                for filename, info in batch_files:
                    try:
                        print(f"  📹 {filename} 처리 중...")
                        landmarks = extract_landmarks_with_holistic(
                            info["path"], holistic
                        )
                        if not landmarks:
                            print(f"    ⚠️ 랜드마크 추출 실패: {filename}")
                            continue

                        processed_sequence = enhanced_preprocess_landmarks(landmarks)
                        if processed_sequence.shape != (TARGET_SEQ_LENGTH, 675):
                            print(
                                f"    ⚠️ 시퀀스 형태 불일치: {filename} - {processed_sequence.shape}"
                            )
                            continue

                        batch_data.append(
                            {
                                "sequence": processed_sequence,
                                "label": info["label"],
                                "filename": filename,
                            }
                        )
                        print(f"    ✅ 성공: {filename}")

                    except Exception as e:
                        print(f"    ❌ 오류: {filename} - {e}")
                        continue

                print(f"✅ 배치 {i//batch_size + 1} 완료: {len(batch_data)}개 성공")
                yield batch_data

    except Exception as e:
        print(f"❌ MediaPipe 처리 중 오류: {e}")
        yield []


def augment_sequence_improved(
    sequence,
    noise_level=AUGMENTATION_NOISE_LEVEL,
    scale_range=AUGMENTATION_SCALE_RANGE,
    rotation_range=AUGMENTATION_ROTATION_RANGE,
):
    """개선된 시퀀스 증강."""
    augmented = sequence.copy()

    # 노이즈 추가
    noise = np.random.normal(0, noise_level, augmented.shape)
    augmented += noise

    # 스케일링
    scale_factor = np.random.uniform(1 - scale_range, 1 + scale_range)
    augmented *= scale_factor

    # 시간축에서의 회전 (시프트)
    shift = np.random.randint(-3, 4)
    if shift > 0:
        augmented = np.roll(augmented, shift, axis=0)
    elif shift < 0:
        augmented = np.roll(augmented, shift, axis=0)

    return augmented


def save_label_cache(label, data):
    """라벨별 데이터를 캐시에 저장합니다."""
    cache_path = get_label_cache_path(label)
    
    # 캐시 디렉토리가 없으면 생성
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # 캐시에 저장할 데이터와 파라미터 정보
    cache_data = {
        "data": data,
        "parameters": {
            "TARGET_SEQ_LENGTH": TARGET_SEQ_LENGTH,
            "AUGMENTATIONS_PER_VIDEO": AUGMENTATIONS_PER_VIDEO,
            "AUGMENTATION_NOISE_LEVEL": AUGMENTATION_NOISE_LEVEL,
            "AUGMENTATION_SCALE_RANGE": AUGMENTATION_SCALE_RANGE,
            "AUGMENTATION_ROTATION_RANGE": AUGMENTATION_ROTATION_RANGE,
            "NONE_CLASS_NOISE_LEVEL": NONE_CLASS_NOISE_LEVEL,
            "NONE_CLASS_AUGMENTATIONS_PER_FRAME": NONE_CLASS_AUGMENTATIONS_PER_FRAME,
            # 데이터 개수 관련 파라미터 추가
            "LABEL_MAX_SAMPLES_PER_CLASS": LABEL_MAX_SAMPLES_PER_CLASS,
            "MIN_SAMPLES_PER_CLASS": MIN_SAMPLES_PER_CLASS,
        },
    }

    # 임시 파일에 먼저 저장 (원자적 쓰기)
    temp_path = cache_path + ".tmp"

    try:
        with open(temp_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 성공적으로 저장되면 최종 위치로 이동
        os.replace(temp_path, cache_path)
        print(f"💾 {label} 라벨 데이터 캐시 저장: {cache_path} ({len(data)}개 샘플)")

    except Exception as e:
        # 오류 발생 시 임시 파일 정리
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

# 라벨별 캐시 파일 경로 생성 함수
def get_label_cache_path(label):
    """라벨별 캐시 파일 경로를 반환합니다. 주요 파라미터를 파일명에 포함시켜 캐시 무효화가 자동으로 되도록 합니다."""
    safe_label = label.replace(" ", "_").replace("/", "_")

    # 데이터 개수 관련 파라미터들을 파일명에 포함
    max_samples_str = (
        f"max{LABEL_MAX_SAMPLES_PER_CLASS}"
        if LABEL_MAX_SAMPLES_PER_CLASS
        else "maxNone"
    )
    min_samples_str = f"min{MIN_SAMPLES_PER_CLASS}"

    return os.path.join(
        CACHE_DIR,
        f"{safe_label}_seq{TARGET_SEQ_LENGTH}_aug{AUGMENTATIONS_PER_VIDEO}_{max_samples_str}_{min_samples_str}.pkl",
    )



# 메인 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미디어파이프 시퀀스 추출 및 S3 업로드 통합 파이프라인")
    parser.add_argument('--config', type=str, default='spec.json', help='설정 파일(JSON) 경로 (기본값: spec.json)')
    parser.add_argument('--s3-bucket', type=str, default='waterandfish-s3', help='S3 버킷 이름 (기본값: waterandfish-s3)')
    parser.add_argument('--s3-prefix', type=str, default='feature-extraction-cache', help='S3 업로드 경로 prefix (기본값: feature-extraction-cache)')
    parser.add_argument('--region', type=str, default='ap-northeast-2', help='S3 리전 (기본값: ap-northeast-2)')
    parser.add_argument('--upload', action='store_true', default=True, help='S3 업로드 실행 (기본값: True)')
    args = parser.parse_args()
    uploader = S3StreamingUploader(args.s3_bucket, args.s3_prefix, args.region)

    """메인 실행 함수"""
    # args.config를 사용하여 설정 파일 읽기
    with open(args.config, "r") as f:
        params = json.load(f)
    label_dict = params["label_dict"]

    ACTIONS = list(label_dict.keys())
    NONE_CLASS = ACTIONS[-1]

    print(f"🔧 라벨 목록: {ACTIONS}")
    # 1. 비디오 루트 디렉토리 검증
    valid_roots = validate_video_roots()
    if not valid_roots:
        print("❌ 유효한 비디오 루트 디렉토리가 없습니다.")
        sys.exit(1)

    # 2. labels.csv 파일 읽기 및 검증
    if not os.path.exists("labels.csv"):
        print("❌ labels.csv 파일이 없습니다.")
        sys.exit(1)

    labels_df = pd.read_csv("labels.csv")
    print(f"📊 labels.csv 로드 완료: {len(labels_df)}개 항목")
    print(labels_df.head())

    # 3. 파일명에서 비디오 루트 경로 추출 (개선된 방식)
    print("\n🔍 파일명 분석 및 경로 매핑 중...")
    file_mapping = {}
    found_files = 0
    missing_files = 0
    filtered_files = 0

    # 라벨별로 파일을 모아서 최대 개수만큼만 샘플링
    label_to_files = defaultdict(list)
    for idx, row in labels_df.iterrows():
        filename = row["파일명"]
        label = row["한국어"]
        if label not in ACTIONS:
            continue
        file_path = get_video_root_and_path(filename)
        if file_path:
            label_to_files[label].append((filename, file_path))
            found_files += 1
            filtered_files += 1
        else:
            missing_files += 1

    # 최대 개수만큼만 샘플링
    for label in ACTIONS:
        files = label_to_files[label]
        if LABEL_MAX_SAMPLES_PER_CLASS is not None:
            files = files[:LABEL_MAX_SAMPLES_PER_CLASS]
        for filename, file_path in files:
            file_mapping[filename] = {"path": file_path, "label": label}

    # [수정] 라벨별 원본 영상 개수 체크 및 최소 개수 미달 시 학습 중단 (None은 예외)
    insufficient_labels = []
    for label in ACTIONS:
        if label == NONE_CLASS:
            continue  # None 클래스는 예외
        num_samples = len(label_to_files[label])
        if num_samples < MIN_SAMPLES_PER_CLASS:
            insufficient_labels.append((label, num_samples))
    if insufficient_labels:
        print("\n❌ 최소 샘플 개수 미달 라벨 발견! 학습을 중단합니다.")
        for label, count in insufficient_labels:
            print(f"   - {label}: {count}개 (최소 필요: {MIN_SAMPLES_PER_CLASS}개)")
        sys.exit(1)

    print(f"\n📊 파일 매핑 결과:")
    print(f"   ✅ 찾은 파일: {found_files}개")
    print(f"   ❌ 누락된 파일: {missing_files}개")
    print(f"   🎯 ACTIONS 라벨에 해당하는 파일: {filtered_files}개")
    print(f"   ⚡ 라벨별 최대 {LABEL_MAX_SAMPLES_PER_CLASS}개 파일만 사용")
    print(f"   ⚡ 라벨별 최소 {MIN_SAMPLES_PER_CLASS}개 파일 필요")

    if len(file_mapping) == 0:
        print("❌ 찾을 수 있는 파일이 없습니다.")
        sys.exit(1)

    # 4. 라벨별 데이터 추출 및 캐싱 (개별 처리)
    print("\n🚀 라벨별 데이터 추출 및 캐싱 시작...")

    # None 클래스 제외한 다른 클래스들의 평균 개수 계산
    other_class_counts = {}
    for filename, info in file_mapping.items():
        if info["label"] != NONE_CLASS:
            label = info["label"]
            other_class_counts[label] = other_class_counts.get(label, 0) + 1

    if other_class_counts:
        avg_other_class_count = sum(other_class_counts.values()) / len(
            other_class_counts
        )
        target_none_count = int(avg_other_class_count * (1 + AUGMENTATIONS_PER_VIDEO))
        print(
            f"📊 다른 클래스 평균: {avg_other_class_count:.1f}개 → None 클래스 목표: {target_none_count}개"
        )
    else:
        target_none_count = None
        print(f"📊 다른 클래스가 없음 → None 클래스 기본값 사용")

    X = []
    y = []

    for label in ACTIONS:
        print(f"\n{'='*50}")
        print(f"📋 {label} 라벨 처리 중...")
        print(f"{'='*50}")

        if label == NONE_CLASS:
            label_data = generate_balanced_none_class_data(
                file_mapping, NONE_CLASS, target_none_count
            )
        else:
            label_data = extract_and_cache_label_data_optimized(file_mapping, label)

        if label_data:
            label_index = get_action_index(label, ACTIONS)
            X.extend(label_data)
            y.extend([label_index] * len(label_data))
            print(f"✅ {label}: {len(label_data)}개 샘플 추가됨")
        else:
            print(f"⚠️ {label}: 데이터가 없습니다.")

    print(f"\n{'='*50}")
    print(f"📊 최종 데이터 통계:")
    print(f"{'='*50}")
    print(f"총 샘플 수: {len(X)}")

    # 클래스별 샘플 수 확인
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if 0 <= class_idx < len(ACTIONS):
            print(f"클래스 {class_idx} ({ACTIONS[class_idx]}): {count}개")
        else:
            print(f"클래스 {class_idx} (Unknown): {count}개")

    # S3 업로드
    if args.upload:
        s3_key = f"{args.s3_prefix}/{label}.pkl.gz"
        print(f"S3 업로드: {s3_key}")
        uploader.upload_pickle_gzip(label_data, s3_key)
        print(f"✅ S3 업로드 완료: {s3_key}")

    print("파이프라인 완료!") 