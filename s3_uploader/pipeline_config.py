#!/usr/bin/env python3
"""
미디어파이프 S3 파이프라인 설정 파일
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

class PipelineConfig:
    """파이프라인 설정 관리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        설정 초기화
        
        Args:
            config_path: 설정 파일 경로 (None이면 기본값 사용)
        """
        self.config_path = config_path
        
        # 기본 설정
        self.default_config = {
            # 비디오 처리 설정
            'video': {
                'target_fps': 30,
                'max_frames': None,  # None이면 전체 프레임 처리
                'video_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'],
                'supported_formats': ['mp4', 'avi', 'mov', 'mkv']
            },
            
            # 미디어파이프 설정
            'mediapipe': {
                'model_complexity': 1,  # 0, 1, 2
                'static_image_mode': False,
                'smooth_landmarks': True,
                'enable_segmentation': False,
                'smooth_segmentation': True,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5
            },
            
            # 출력 설정
            'output': {
                'compression': True,
                'format': 'pkl.gz',  # pickle + gzip
                'naming_convention': '{video_name}_pose_sequence.{ext}'
            },
            
            # S3 설정
            's3': {
                'region': 'us-east-1',
                'max_workers': 4,
                'chunk_size': 8 * 1024 * 1024,  # 8MB
                'overwrite': False,
                'use_multipart': True,
                'multipart_threshold': 100 * 1024 * 1024  # 100MB
            },
            
            # 성능 설정
            'performance': {
                'batch_size': 1,  # 동시 처리 비디오 수
                'memory_limit_gb': 8,  # 메모리 제한
                'cpu_workers': 4,  # CPU 워커 수
                'gpu_memory_fraction': 0.8  # GPU 메모리 사용 비율
            },
            
            # 로깅 설정
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': True,
                'log_dir': 'logs'
            }
        }
        
        # 사용자 설정 로드
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """설정 로드"""
        if self.config_path and os.path.exists(self.config_path):
            import json
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 기본 설정과 사용자 설정 병합
            config = self.merge_configs(self.default_config, user_config)
        else:
            config = self.default_config.copy()
        
        return config
    
    def merge_configs(self, default: Dict, user: Dict) -> Dict:
        """설정 병합 (재귀적)"""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config_path: Optional[str] = None) -> str:
        """설정 저장"""
        if config_path is None:
            config_path = self.config_path or 'pipeline_config.json'
        
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def get(self, key_path: str, default=None):
        """중첩된 키로 설정값 가져오기"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """중첩된 키로 설정값 설정"""
        keys = key_path.split('.')
        config = self.config
        
        # 마지막 키까지 탐색
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 마지막 키 설정
        config[keys[-1]] = value
    
    def validate_config(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        # 비디오 설정 검사
        target_fps = self.get('video.target_fps', 0)
        if target_fps is not None and target_fps <= 0:
            errors.append("target_fps는 0보다 커야 합니다")
        
        max_frames = self.get('video.max_frames')
        if max_frames is not None and max_frames < 0:
            errors.append("max_frames는 0 이상이어야 합니다")
        
        # 미디어파이프 설정 검사
        model_complexity = self.get('mediapipe.model_complexity', 1)
        if model_complexity not in [0, 1, 2]:
            errors.append("model_complexity는 0, 1, 2 중 하나여야 합니다")
        
        detection_conf = self.get('mediapipe.min_detection_confidence', 0.5)
        tracking_conf = self.get('mediapipe.min_tracking_confidence', 0.5)
        
        if not (0 <= detection_conf <= 1):
            errors.append("min_detection_confidence는 0과 1 사이여야 합니다")
        
        if not (0 <= tracking_conf <= 1):
            errors.append("min_tracking_confidence는 0과 1 사이여야 합니다")
        
        # S3 설정 검사
        if self.get('s3.max_workers', 0) <= 0:
            errors.append("max_workers는 0보다 커야 합니다")
        
        if self.get('s3.chunk_size', 0) <= 0:
            errors.append("chunk_size는 0보다 커야 합니다")
        
        return errors
    
    def get_extractor_kwargs(self) -> Dict:
        """미디어파이프 추출기 인자 반환"""
        return {
            'model_complexity': self.get('mediapipe.model_complexity', 1),
            'static_image_mode': self.get('mediapipe.static_image_mode', False),
            'smooth_landmarks': self.get('mediapipe.smooth_landmarks', True),
            'enable_segmentation': self.get('mediapipe.enable_segmentation', False),
            'smooth_segmentation': self.get('mediapipe.smooth_segmentation', True),
            'min_detection_confidence': self.get('mediapipe.min_detection_confidence', 0.5),
            'min_tracking_confidence': self.get('mediapipe.min_tracking_confidence', 0.5)
        }
    
    def get_uploader_kwargs(self) -> Dict:
        """S3 업로더 인자 반환"""
        return {
            'max_workers': self.get('s3.max_workers', 4),
            'chunk_size': self.get('s3.chunk_size', 8 * 1024 * 1024)
        }
    
    def get_video_kwargs(self) -> Dict:
        """비디오 처리 인자 반환"""
        return {
            'target_fps': self.get('video.target_fps', 30),
            'max_frames': self.get('video.max_frames'),
            'video_extensions': self.get('video.video_extensions', ['.mp4', '.avi', '.mov', '.mkv'])
        }

# 환경별 설정 프리셋
class ConfigPresets:
    """설정 프리셋"""
    
    @staticmethod
    def development() -> Dict:
        """개발 환경 설정"""
        return {
            'video': {
                'target_fps': 15,
                'max_frames': 300  # 10초 (15fps)
            },
            'mediapipe': {
                'model_complexity': 0,  # 빠른 처리
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.3
            },
            's3': {
                'max_workers': 2
            },
            'performance': {
                'batch_size': 1,
                'memory_limit_gb': 4
            }
        }
    
    @staticmethod
    def production() -> Dict:
        """프로덕션 환경 설정"""
        return {
            'video': {
                'target_fps': 30,
                'max_frames': None  # 전체 프레임
            },
            'mediapipe': {
                'model_complexity': 2,  # 높은 정확도
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.7
            },
            's3': {
                'max_workers': 8,
                'chunk_size': 16 * 1024 * 1024  # 16MB
            },
            'performance': {
                'batch_size': 2,
                'memory_limit_gb': 16
            }
        }
    
    @staticmethod
    def high_accuracy() -> Dict:
        """고정확도 설정"""
        return {
            'mediapipe': {
                'model_complexity': 2,
                'min_detection_confidence': 0.8,
                'min_tracking_confidence': 0.8,
                'smooth_landmarks': True
            },
            'video': {
                'target_fps': 60  # 높은 FPS
            }
        }
    
    @staticmethod
    def fast_processing() -> Dict:
        """빠른 처리 설정"""
        return {
            'mediapipe': {
                'model_complexity': 0,
                'min_detection_confidence': 0.3,
                'min_tracking_confidence': 0.3
            },
            'video': {
                'target_fps': 15,
                'max_frames': 150  # 10초
            },
            's3': {
                'max_workers': 12
            }
        }

def create_config_file(config_path: str = 'pipeline_config.json', preset: str = 'development'):
    """설정 파일 생성"""
    config = PipelineConfig()
    
    # 프리셋 적용
    if preset == 'development':
        preset_config = ConfigPresets.development()
    elif preset == 'production':
        preset_config = ConfigPresets.production()
    elif preset == 'high_accuracy':
        preset_config = ConfigPresets.high_accuracy()
    elif preset == 'fast_processing':
        preset_config = ConfigPresets.fast_processing()
    else:
        raise ValueError(f"알 수 없는 프리셋: {preset}")
    
    # 프리셋 설정 적용
    for key, value in preset_config.items():
        if key in config.config:
            config.config[key].update(value)
        else:
            config.config[key] = value
    
    # 설정 저장
    config.save_config(config_path)
    print(f"설정 파일이 생성되었습니다: {config_path}")
    
    return config_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='파이프라인 설정 파일 생성')
    parser.add_argument('--output', '-o', default='pipeline_config.json', help='출력 파일 경로')
    parser.add_argument('--preset', '-p', default='development', 
                       choices=['development', 'production', 'high_accuracy', 'fast_processing'],
                       help='설정 프리셋')
    
    args = parser.parse_args()
    
    create_config_file(args.output, args.preset) 