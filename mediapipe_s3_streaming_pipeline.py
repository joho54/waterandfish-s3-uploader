#!/usr/bin/env python3
"""
미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 파이프라인
로컬 저장 없이 메모리에서 직접 S3로 업로드
"""

import os
import json
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
import io
import pickle
import gzip
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import boto3
from tqdm import tqdm
import mediapipe as mp

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MediaPipeStreamingExtractor:
    """수어 인식을 위한 최적화된 랜드마크 추출 클래스 (pose, left_hand, right_hand)"""
    
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.3,  # 더 관대한 설정으로 변경
                 min_tracking_confidence=0.3):  # 더 관대한 설정으로 변경
        """
        MediaPipe Holistic 초기화 (수어 인식 최적화)
        
        Args:
            static_image_mode: 정적 이미지 모드
            model_complexity: 모델 복잡도 (0, 1, 2)
            smooth_landmarks: 랜드마크 스무딩
            enable_segmentation: 세그멘테이션 활성화
            smooth_segmentation: 세그멘테이션 스무딩
            min_detection_confidence: 최소 검출 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        self.mp_holistic = mp.solutions.holistic
        
        # MediaPipe 설정에 InputStreamHandler 추가
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 타임스탬프 문제 해결을 위한 설정
        mp.solutions.drawing_utils.DrawingSpec = mp.solutions.drawing_utils.DrawingSpec
        
        # 수어 인식을 위한 핵심 랜드마크 인덱스 정의
        self.pose_landmark_indices = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        # 손 랜드마크는 21개씩 (MediaPipe Hand의 기본)
        self.hand_landmark_indices = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20
        }
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        프레임에서 수어 인식을 위한 랜드마크 추출 (pose, left_hand, right_hand)
        
        Args:
            frame: 입력 프레임 (BGR)
            
        Returns:
            랜드마크 딕셔너리 또는 None
        """
        try:
            # BGR을 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Holistic 검출
            results = self.holistic.process(rgb_frame)
            
            frame_data = {}
            
            # 포즈 랜드마크 추출
            if results.pose_landmarks:
                pose_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    pose_landmarks.append([landmark.x, landmark.y, landmark.visibility])
                frame_data['pose'] = np.array(pose_landmarks)
            else:
                frame_data['pose'] = None
            
            # 왼손 랜드마크 추출
            if results.left_hand_landmarks:
                left_hand_landmarks = []
                for landmark in results.left_hand_landmarks.landmark:
                    left_hand_landmarks.append([landmark.x, landmark.y, landmark.z])
                frame_data['left_hand'] = np.array(left_hand_landmarks)
            else:
                frame_data['left_hand'] = None
            
            # 오른손 랜드마크 추출
            if results.right_hand_landmarks:
                right_hand_landmarks = []
                for landmark in results.right_hand_landmarks.landmark:
                    right_hand_landmarks.append([landmark.x, landmark.y, landmark.z])
                frame_data['right_hand'] = np.array(right_hand_landmarks)
            else:
                frame_data['right_hand'] = None
            
            # 최소한 포즈나 손 중 하나라도 검출되었으면 반환
            if frame_data['pose'] is not None or frame_data['left_hand'] is not None or frame_data['right_hand'] is not None:
                return frame_data
            
            return None
            
        except Exception as e:
            # MediaPipe 타임스탬프 오류 등은 무시하고 계속 진행
            if "timestamp mismatch" in str(e).lower() or "graph has errors" in str(e).lower():
                return None
            else:
                # 다른 오류는 로깅
                logger.warning(f"랜드마크 추출 중 오류: {e}")
                return None
    
    def extract_sequence_to_memory(self, 
                                 video_path: str,
                                 target_fps: int = 30,
                                 max_frames: Optional[int] = None) -> Tuple[bytes, Dict]:
        """
        비디오에서 수어 인식을 위한 랜드마크 시퀀스를 메모리에 추출
        
        Args:
            video_path: 입력 비디오 경로
            target_fps: 목표 FPS (다운샘플링)
            max_frames: 최대 프레임 수
            
        Returns:
            (압축된 시퀀스 데이터, 메타데이터)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
        
        try:
            # 비디오 정보
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 비디오 정보 유효성 검사
            if total_frames <= 0:
                raise ValueError(f"유효하지 않은 프레임 수: {total_frames}")
            if original_fps <= 0:
                raise ValueError(f"유효하지 않은 FPS: {original_fps}")
            
            duration = total_frames / original_fps
            
            logger.info(f"비디오 정보: {total_frames} 프레임, {original_fps:.2f} FPS, {duration:.2f}초")
            
            # 프레임 간격 계산
            frame_interval = max(1, int(original_fps / target_fps))
            
            # 시퀀스 저장용 리스트
            landmark_sequences = []
            frame_timestamps = []
            extracted_frames = 0
            
            # 통계 추적
            pose_detected = 0
            left_hand_detected = 0
            right_hand_detected = 0
            failed_frames = 0
            
            # 진행률 표시
            pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames,
                       desc=f"수어 랜드마크 추출 중: {Path(video_path).name}")
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 간격에 따라 추출
                if frame_idx % frame_interval == 0:
                    try:
                        landmarks = self.extract_landmarks(frame)
                        
                        if landmarks is not None:
                            landmark_sequences.append(landmarks)
                            frame_timestamps.append(frame_idx / original_fps)
                            extracted_frames += 1
                            
                            # 통계 업데이트
                            if landmarks['pose'] is not None:
                                pose_detected += 1
                            if landmarks['left_hand'] is not None:
                                left_hand_detected += 1
                            if landmarks['right_hand'] is not None:
                                right_hand_detected += 1
                        else:
                            failed_frames += 1
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        # MediaPipe 타임스탬프 오류는 무시
                        if "timestamp mismatch" in error_msg or "graph has errors" in error_msg:
                            failed_frames += 1
                        else:
                            logger.warning(f"프레임 {frame_idx} 처리 실패: {e}")
                            failed_frames += 1
                    
                    pbar.update(1)
                    
                    # 최대 프레임 수 체크
                    if max_frames and extracted_frames >= max_frames:
                        break
                
                frame_idx += 1
            
            pbar.close()
            
            # 최소한의 랜드마크가 추출되었는지 확인
            if len(landmark_sequences) == 0:
                raise ValueError(f"랜드마크가 추출되지 않았습니다. 총 {total_frames} 프레임 중 {failed_frames} 프레임 실패")
            
            # 성공률이 너무 낮으면 경고 (하지만 계속 진행)
            success_rate = len(landmark_sequences) / (len(landmark_sequences) + failed_frames) if (len(landmark_sequences) + failed_frames) > 0 else 0
            if success_rate < 0.1:  # 10% 미만이면 경고
                logger.warning(f"랜드마크 추출 성공률이 낮습니다: {success_rate:.1%} ({len(landmark_sequences)}/{len(landmark_sequences) + failed_frames})")
            
            # 결과 데이터 구성
            result_data = {
                'video_path': video_path,
                'landmark_sequences': landmark_sequences,
                'frame_timestamps': np.array(frame_timestamps),
                'metadata': {
                    'total_frames': total_frames,
                    'extracted_frames': len(landmark_sequences),
                    'original_fps': original_fps,
                    'target_fps': target_fps,
                    'frame_interval': frame_interval,
                    'duration': duration,
                    'pose_landmark_count': 33,
                    'hand_landmark_count': 21,
                    'pose_landmark_indices': self.pose_landmark_indices,
                    'hand_landmark_indices': self.hand_landmark_indices,
                    'detection_stats': {
                        'pose_detected': pose_detected,
                        'left_hand_detected': left_hand_detected,
                        'right_hand_detected': right_hand_detected,
                        'pose_detection_rate': pose_detected / len(landmark_sequences) if landmark_sequences else 0,
                        'left_hand_detection_rate': left_hand_detected / len(landmark_sequences) if landmark_sequences else 0,
                        'right_hand_detection_rate': right_hand_detected / len(landmark_sequences) if landmark_sequences else 0,
                        'failed_frames': failed_frames
                    }
                }
            }
            
            # 메모리에서 압축
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
                pickle.dump(result_data, f)
            
            compressed_data = buffer.getvalue()
            
            logger.info(f"수어 랜드마크 시퀀스 추출 완료: {len(landmark_sequences)} 프레임 -> {len(compressed_data) / (1024*1024):.2f} MB")
            logger.info(f"검출 통계 - 포즈: {pose_detected}, 왼손: {left_hand_detected}, 오른손: {right_hand_detected}, 실패: {failed_frames}")
            
            return compressed_data, {
                'video_path': video_path,
                'extracted_frames': len(landmark_sequences),
                'total_frames': total_frames,
                'compression_ratio': len(landmark_sequences) / total_frames,
                'file_size_mb': len(compressed_data) / (1024 * 1024),
                'detection_stats': {
                    'pose_detected': pose_detected,
                    'left_hand_detected': left_hand_detected,
                    'right_hand_detected': right_hand_detected,
                    'failed_frames': failed_frames
                }
            }
            
        finally:
            cap.release()
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'holistic'):
            self.holistic.close()

class S3StreamingUploader:
    """메모리에서 직접 S3로 스트리밍 업로드하는 클래스"""
    
    def __init__(self, 
                 bucket_name: str,
                 region_name: str = 'us-east-1',
                 max_workers: int = 4,
                 chunk_size: int = 8 * 1024 * 1024):  # 8MB 청크
        """
        S3 스트리밍 업로더 초기화
        
        Args:
            bucket_name: S3 버킷 이름
            region_name: AWS 리전
            max_workers: 동시 업로드 스레드 수
            chunk_size: 멀티파트 업로드 청크 크기
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        
        # S3 클라이언트 초기화
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        # 업로드 진행률 추적
        self.upload_lock = threading.Lock()
        self.upload_stats = {
            'total_files': 0,
            'uploaded_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_size': 0,
            'uploaded_size': 0
        }
    
    def check_file_exists_simple(self, s3_key: str) -> bool:
        """S3에 파일이 존재하는지 간단히 확인 (ETag 비교 없이)"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def calculate_data_hash(self, data: bytes) -> str:
        """데이터의 MD5 해시 계산"""
        return hashlib.md5(data).hexdigest()
    
    def check_file_exists(self, s3_key: str, data_hash: str) -> bool:
        """S3에 파일이 존재하는지 확인 (해시 비교)"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            etag = response.get('ETag', '').strip('"')
            return etag == data_hash
        except:
            return False
    
    def upload_data_streaming(self, 
                            data: bytes, 
                            s3_key: str,
                            overwrite: bool = False) -> Dict:
        """
        메모리 데이터를 S3에 스트리밍 업로드
        
        Args:
            data: 업로드할 데이터 (bytes)
            s3_key: S3 키
            overwrite: 기존 파일 덮어쓰기 여부
            
        Returns:
            업로드 결과
        """
        try:
            data_size = len(data)
            
            # 데이터 해시 계산
            data_hash = self.calculate_data_hash(data)
            
            # 기존 파일 확인
            if not overwrite and self.check_file_exists(s3_key, data_hash):
                logger.info(f"파일이 이미 존재합니다: {s3_key}")
                
                with self.upload_lock:
                    self.upload_stats['skipped_files'] += 1
                
                return {
                    'status': 'skipped',
                    's3_key': s3_key,
                    'file_size': data_size,
                    'reason': 'file_already_exists'
                }
            
            # 업로드 실행
            start_time = time.time()
            
            if data_size > self.chunk_size:
                # 멀티파트 업로드
                result = self._multipart_upload_data(data, s3_key, data_size)
            else:
                # 단일 업로드
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=data
                )
                result = {'status': 'success'}
            
            upload_time = time.time() - start_time
            speed = data_size / upload_time if upload_time > 0 else 0
            
            # 통계 업데이트
            with self.upload_lock:
                self.upload_stats['uploaded_files'] += 1
                self.upload_stats['uploaded_size'] += data_size
            
            return {
                'status': 'success',
                's3_key': s3_key,
                'file_size': data_size,
                'upload_time': upload_time,
                'speed_mbps': speed / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"업로드 실패 {s3_key}: {e}")
            
            with self.upload_lock:
                self.upload_stats['failed_files'] += 1
            
            return {
                'status': 'failed',
                's3_key': s3_key,
                'error': str(e)
            }
    
    def _multipart_upload_data(self, data: bytes, s3_key: str, data_size: int) -> Dict:
        """메모리 데이터 멀티파트 업로드"""
        try:
            # 멀티파트 업로드 시작
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            upload_id = response['UploadId']
            
            parts = []
            part_number = 1
            
            # 데이터를 청크로 분할
            for i in range(0, data_size, self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                
                # 파트 업로드
                part_response = self.s3_client.upload_part(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk
                )
                
                parts.append({
                    'ETag': part_response['ETag'],
                    'PartNumber': part_number
                })
                
                part_number += 1
            
            # 멀티파트 업로드 완료
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            return {'status': 'success'}
            
        except Exception as e:
            # 업로드 실패 시 정리
            try:
                self.s3_client.abort_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    UploadId=upload_id
                )
            except:
                pass
            raise e

class MediaPipeS3StreamingPipeline:
    """미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 통합 파이프라인"""
    
    def __init__(self, 
                 video_dir: str,
                 s3_bucket: str,
                 s3_prefix: str,
                 aws_region: str = 'us-east-1',
                 max_workers: int = 4,
                 **extractor_kwargs):
        """
        스트리밍 파이프라인 초기화
        
        Args:
            video_dir: 비디오 디렉토리
            s3_bucket: S3 버킷 이름
            s3_prefix: S3 접두사
            aws_region: AWS 리전
            max_workers: 동시 처리 스레드 수
            **extractor_kwargs: 추출기 설정
        """
        self.video_dir = Path(video_dir)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.aws_region = aws_region
        self.max_workers = max_workers
        
        # 컴포넌트 초기화
        self.extractor = MediaPipeStreamingExtractor(**extractor_kwargs)
        self.uploader = S3StreamingUploader(
            bucket_name=s3_bucket,
            region_name=aws_region,
            max_workers=max_workers
        )
        
        # 파이프라인 상태
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'extraction_results': [],
            'upload_results': [],
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0
        }
    
    def process_video_streaming(self, 
                              video_path: Path,
                              target_fps: int = 30,
                              max_frames: Optional[int] = None,
                              skip_existing: bool = True) -> Dict:
        """
        단일 비디오를 스트리밍 처리 (추출 + 업로드)
        
        Args:
            video_path: 비디오 파일 경로
            target_fps: 목표 FPS
            max_frames: 최대 프레임 수
            skip_existing: 기존 파일 건너뛰기 여부
            
        Returns:
            처리 결과
        """
        try:
            # 파일 존재 여부 확인
            if not video_path.exists():
                logger.error(f"비디오 파일이 존재하지 않습니다: {video_path}")
                return {
                    'video_path': str(video_path),
                    'error': 'file_not_found',
                    'status': 'failed'
                }
            
            # 파일 크기 확인
            file_size = video_path.stat().st_size
            if file_size == 0:
                logger.error(f"비디오 파일이 비어있습니다: {video_path}")
                return {
                    'video_path': str(video_path),
                    'error': 'empty_file',
                    'status': 'failed'
                }
            
            # S3 키 생성
            s3_key = f"{self.s3_prefix}/{video_path.stem}_sign_language_landmarks.pkl.gz"
            
            # 기존 파일 확인 (조기 건너뛰기)
            if skip_existing and self.uploader.check_file_exists_simple(s3_key):
                logger.info(f"이미 처리된 파일입니다: {video_path.name} -> {s3_key}")
                return {
                    'video_path': str(video_path),
                    's3_key': s3_key,
                    'status': 'skipped',
                    'reason': 'file_already_exists_in_s3'
                }
            
            logger.info(f"처리 시작: {video_path.name} ({file_size / (1024*1024):.2f} MB)")
            
            # 1단계: 메모리에서 시퀀스 추출
            compressed_data, extraction_info = self.extractor.extract_sequence_to_memory(
                str(video_path),
                target_fps=target_fps,
                max_frames=max_frames
            )
            
            # 추출된 프레임 수 확인
            if extraction_info['extracted_frames'] == 0:
                logger.warning(f"랜드마크가 추출되지 않았습니다: {video_path}")
                return {
                    'video_path': str(video_path),
                    'error': 'no_landmarks_detected',
                    'status': 'failed'
                }
            
            # 2단계: S3에 스트리밍 업로드
            upload_result = self.uploader.upload_data_streaming(
                compressed_data,
                s3_key,
                overwrite=False
            )
            
            # 결과 통합
            result = {
                'video_path': str(video_path),
                's3_key': s3_key,
                'extraction_info': extraction_info,
                'upload_result': upload_result,
                'status': upload_result['status']
            }
            
            logger.info(f"처리 완료: {video_path.name} -> {extraction_info['extracted_frames']} 프레임")
            return result
            
        except Exception as e:
            logger.error(f"비디오 처리 실패 {video_path}: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'video_path': str(video_path),
                'error': str(e),
                'status': 'failed'
            }
    
    def run_streaming_pipeline(self, 
                             target_fps: int = 30,
                             max_frames: Optional[int] = None,
                             skip_existing: bool = True,
                             video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']) -> Dict:
        """
        스트리밍 파이프라인 실행
        
        Args:
            target_fps: 목표 FPS
            max_frames: 최대 프레임 수
            video_extensions: 지원하는 비디오 확장자
            
        Returns:
            파이프라인 실행 결과
        """
        self.pipeline_stats['start_time'] = time.time()
        
        logger.info("=== 수어 인식용 미디어파이프 S3 스트리밍 파이프라인 시작 ===")
        logger.info(f"비디오 디렉토리: {self.video_dir}")
        logger.info(f"S3 버킷: {self.s3_bucket}")
        logger.info(f"S3 접두사: {self.s3_prefix}")
        logger.info(f"동시 처리 스레드: {self.max_workers}개")
        logger.info(f"기존 파일 건너뛰기: {skip_existing}")
        
        # 비디오 파일 찾기
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.video_dir.glob(f"*{ext}"))
            video_files.extend(self.video_dir.glob(f"*{ext.upper()}"))
        
        # 중복 제거 및 정렬
        video_files = sorted(list(set(video_files)))
        
        # 파일 유효성 사전 검사
        valid_video_files = []
        for video_file in video_files:
            try:
                # 파일 존재 및 크기 확인
                if not video_file.exists():
                    logger.warning(f"파일이 존재하지 않습니다: {video_file}")
                    continue
                
                file_size = video_file.stat().st_size
                if file_size == 0:
                    logger.warning(f"빈 파일입니다: {video_file}")
                    continue
                
                # 최소 크기 확인 (1KB)
                if file_size < 1024:
                    logger.warning(f"파일이 너무 작습니다: {video_file} ({file_size} bytes)")
                    continue
                
                valid_video_files.append(video_file)
                
            except Exception as e:
                logger.warning(f"파일 검증 실패 {video_file}: {e}")
                continue
        
        self.pipeline_stats['total_videos'] = len(valid_video_files)
        logger.info(f"발견된 비디오 파일: {len(video_files)}개")
        logger.info(f"유효한 비디오 파일: {len(valid_video_files)}개")
        
        if not valid_video_files:
            logger.warning("처리할 유효한 비디오 파일이 없습니다.")
            return {
                'status': 'success',
                'message': 'no_valid_videos_found',
                'statistics': self.pipeline_stats
            }
        
        results = []
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 작업 제출
            future_to_video = {}
            for video_file in valid_video_files:
                future = executor.submit(
                    self.process_video_streaming,
                    video_file,
                    target_fps,
                    max_frames,
                    skip_existing
                )
                future_to_video[future] = video_file
            
            # 결과 수집
            with tqdm(total=len(valid_video_files), desc="스트리밍 처리") as pbar:
                for future in as_completed(future_to_video):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # 진행률 표시 업데이트
                    successful = [r for r in results if r['status'] == 'success']
                    failed = [r for r in results if r['status'] == 'failed']
                    skipped = [r for r in results if r['status'] == 'skipped']
                    
                    pbar.set_postfix({
                        'Success': len(successful),
                        'Failed': len(failed),
                        'Skipped': len(skipped)
                    })
        
        # 결과 분석
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        skipped = [r for r in results if r['status'] == 'skipped']
        
        # 실패 원인 분석
        error_counts = {}
        for result in failed:
            error_type = result.get('error', 'unknown_error')
            if error_type in error_counts:
                error_counts[error_type] += 1
            else:
                error_counts[error_type] = 1
        
        self.pipeline_stats['processed_videos'] = len(successful)
        self.pipeline_stats['failed_videos'] = len(failed)
        self.pipeline_stats['skipped_videos'] = len(skipped)
        self.pipeline_stats['extraction_results'] = results
        self.pipeline_stats['error_analysis'] = error_counts
        
        self.pipeline_stats['end_time'] = time.time()
        
        # 최종 결과 요약
        duration = self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
        logger.info("=== 수어 인식 스트리밍 파이프라인 완료 ===")
        logger.info(f"총 실행 시간: {duration:.2f}초")
        logger.info(f"처리된 비디오: {len(successful)}/{len(valid_video_files)}")
        logger.info(f"실패한 비디오: {len(failed)}개")
        logger.info(f"건너뛴 비디오: {len(skipped)}개")
        
        if skipped:
            logger.info("=== 건너뛴 파일 상세 ===")
            for result in skipped:
                reason = result.get('reason', 'unknown')
                logger.info(f"  {Path(result['video_path']).name}: {reason}")
        
        # 실패 원인 상세 분석
        if error_counts:
            logger.info("=== 실패 원인 분석 ===")
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(failed)) * 100
                logger.info(f"  {error_type}: {count}개 ({percentage:.1f}%)")
        
        if successful:
            total_size = sum(r['extraction_info']['file_size_mb'] for r in successful)
            avg_compression = sum(r['extraction_info']['compression_ratio'] for r in successful) / len(successful)
            logger.info(f"총 업로드 크기: {total_size:.2f} MB")
            logger.info(f"평균 압축률: {avg_compression:.2%}")
        
        return {
            'status': 'success',
            'duration': duration,
            'results': results,
            'statistics': self.pipeline_stats,
            'error_analysis': error_counts
        }

def main():
    parser = argparse.ArgumentParser(description='수어 인식용 미디어파이프 S3 스트리밍 파이프라인')
    parser.add_argument('--video-dir', '-v', required=True, help='비디오 디렉토리')
    parser.add_argument('--s3-bucket', '-b', required=True, help='S3 버킷 이름')
    parser.add_argument('--s3-prefix', '-p', required=True, help='S3 접두사')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS 리전')
    parser.add_argument('--target-fps', type=int, default=30, help='목표 FPS')
    parser.add_argument('--max-frames', type=int, help='최대 프레임 수')
    parser.add_argument('--max-workers', type=int, default=4, help='동시 처리 스레드 수')
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2], help='모델 복잡도')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='S3에 이미 존재하는 파일 건너뛰기')
    parser.add_argument('--force-overwrite', action='store_true', help='기존 파일 덮어쓰기 (--skip-existing 무시)')
    args = parser.parse_args()
    
    # 스트리밍 파이프라인 초기화
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir=args.video_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        aws_region=args.aws_region,
        max_workers=args.max_workers,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.3,  # 더 관대한 설정
        min_tracking_confidence=0.3    # 더 관대한 설정
    )
    
    # skip_existing 설정 결정
    skip_existing = args.skip_existing and not args.force_overwrite
    
    # 스트리밍 파이프라인 실행
    result = pipeline.run_streaming_pipeline(
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        skip_existing=skip_existing
    )
    
    if result['status'] == 'success':
        logger.info("수어 인식 스트리밍 파이프라인이 성공적으로 완료되었습니다!")
    else:
        logger.error(f"수어 인식 스트리밍 파이프라인 실패: {result.get('error', 'Unknown error')}")
        exit(1)

if __name__ == "__main__":
    main() 