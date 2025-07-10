#!/usr/bin/env python3
"""
MediaPipe 테스트 스크립트
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mediapipe_on_video(video_path: str, max_frames: int = 10):
    """비디오에서 MediaPipe 테스트"""
    
    # MediaPipe Holistic 초기화 (매우 관대한 설정)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.1,  # 매우 관대한 설정
        min_tracking_confidence=0.1    # 매우 관대한 설정
    )
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"비디오 정보: {total_frames} 프레임, {fps:.2f} FPS")
        
        frame_count = 0
        detected_frames = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Holistic 검출
            results = holistic.process(rgb_frame)
            
            # 검출 결과 확인
            detected_parts = []
            if results.pose_landmarks:
                detected_parts.append('pose')
            if results.left_hand_landmarks:
                detected_parts.append('left_hand')
            if results.right_hand_landmarks:
                detected_parts.append('right_hand')
            
            if detected_parts:
                detected_frames += 1
                logger.info(f"프레임 {frame_count + 1}: 검출된 부분 - {', '.join(detected_parts)}")
            else:
                logger.warning(f"프레임 {frame_count + 1}: 아무것도 검출되지 않음")
            
            frame_count += 1
        
        success_rate = detected_frames / frame_count if frame_count > 0 else 0
        logger.info(f"테스트 결과: {detected_frames}/{frame_count} 프레임에서 검출됨 (성공률: {success_rate:.1%})")
        
    finally:
        cap.release()
        holistic.close()

def test_mediapipe_on_image(image_path: str):
    """이미지에서 MediaPipe 테스트"""
    
    # MediaPipe Holistic 초기화
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=True,  # 정적 이미지 모드
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    # BGR을 RGB로 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Holistic 검출
    results = holistic.process(rgb_image)
    
    # 검출 결과 확인
    detected_parts = []
    if results.pose_landmarks:
        detected_parts.append('pose')
    if results.left_hand_landmarks:
        detected_parts.append('left_hand')
    if results.right_hand_landmarks:
        detected_parts.append('right_hand')
    
    if detected_parts:
        logger.info(f"이미지에서 검출된 부분: {', '.join(detected_parts)}")
    else:
        logger.warning("이미지에서 아무것도 검출되지 않음")
    
    holistic.close()

if __name__ == "__main__":
    # 테스트할 비디오 파일 경로 (첫 번째 비디오 파일 찾기)
    video_dir = "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/0001~3000(영상)"
    video_files = list(Path(video_dir).glob("*.avi"))
    
    if video_files:
        test_video = str(video_files[0])
        logger.info(f"테스트할 비디오: {test_video}")
        test_mediapipe_on_video(test_video, max_frames=5)
    else:
        logger.error("테스트할 비디오 파일을 찾을 수 없습니다.") 