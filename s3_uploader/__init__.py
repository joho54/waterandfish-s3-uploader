#!/usr/bin/env python3
"""
S3 Uploader Package
미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 패키지
"""

__version__ = "1.0.0"
__author__ = "SaturdayDinner"
__email__ = "your-email@example.com"

# 주요 클래스들을 패키지 레벨에서 import 가능하도록 설정
from .mediapipe_s3_streaming_pipeline import (
    MediaPipeStreamingExtractor,
    S3StreamingUploader,
    MediaPipeS3StreamingPipeline
)

from .pipeline_config import (
    PipelineConfig,
    ConfigPresets,
    create_config_file
)

# 패키지에서 사용할 수 있는 주요 클래스들
__all__ = [
    # 메인 파이프라인 클래스
    "MediaPipeS3StreamingPipeline",
    
    # 개별 컴포넌트
    "MediaPipeStreamingExtractor",
    "S3StreamingUploader",
    
    # 설정 관리
    "PipelineConfig",
    "ConfigPresets",
    "create_config_file",
    
    # 버전 정보
    "__version__",
    "__author__",
    "__email__"
]

# 패키지 초기화 시 로그 메시지
import logging
logger = logging.getLogger(__name__)
logger.info(f"S3 Uploader Package v{__version__} initialized") 