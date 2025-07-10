#!/usr/bin/env python3
"""
S3 Uploader 패키지 사용법 예시
"""

import os
from pathlib import Path
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig

def main():
    """패키지 사용법 예시"""
    print("=== S3 Uploader 패키지 사용법 예시 ===")
    
    # 1. 설정 파일 생성
    print("\n1. 설정 파일 생성 중...")
    config = PipelineConfig()
    config.set('video.target_fps', 30)
    config.set('mediapipe.model_complexity', 1)
    config.set('s3.max_workers', 4)
    
    # 설정 저장
    config.save_config('example_config.json')
    print("   설정 파일 생성: example_config.json")
    
    # 2. 파이프라인 초기화 (예시)
    print("\n2. 파이프라인 초기화 중...")
    print("   실제 사용 시에는 다음 경로들을 수정하세요:")
    print("   - video_dir: 실제 비디오 디렉토리 경로")
    print("   - s3_bucket: 실제 S3 버킷 이름")
    
    # 예시 파이프라인 (실제로는 실행되지 않음)
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir="/path/to/your/videos",  # 실제 경로로 변경
        s3_bucket="your-bucket-name",      # 실제 버킷 이름으로 변경
        s3_prefix="pose-sequences",
        aws_region="us-east-1"
    )
    
    print("   파이프라인 초기화 완료")
    
    # 3. 설정 프리셋 사용법
    print("\n3. 설정 프리셋 사용법:")
    
    # 개발 환경 설정
    dev_config = PipelineConfig()
    dev_config.config.update(PipelineConfig.ConfigPresets.development())
    print("   - 개발 환경: 빠른 테스트용 (15fps, 300프레임)")
    
    # 프로덕션 환경 설정
    prod_config = PipelineConfig()
    prod_config.config.update(PipelineConfig.ConfigPresets.production())
    print("   - 프로덕션 환경: 고품질 처리용 (30fps, 전체 프레임)")
    
    # 4. 개별 컴포넌트 사용법
    print("\n4. 개별 컴포넌트 사용법:")
    
    from s3_uploader import MediaPipeStreamingExtractor, S3StreamingUploader
    
    # 추출기만 사용
    extractor = MediaPipeStreamingExtractor(
        model_complexity=1,
        min_detection_confidence=0.5
    )
    print("   - MediaPipeStreamingExtractor: 시퀀스 추출 전용")
    
    # 업로더만 사용
    uploader = S3StreamingUploader(
        bucket_name="example-bucket",
        region_name="us-east-1"
    )
    print("   - S3StreamingUploader: S3 업로드 전용")
    
    # 5. 실제 사용 시 주의사항
    print("\n5. 실제 사용 시 주의사항:")
    print("   - AWS 자격 증명 설정 필요 (aws configure)")
    print("   - S3 버킷 생성 및 권한 설정 필요")
    print("   - 충분한 디스크 공간 확보")
    print("   - 네트워크 연결 상태 확인")
    
    # 6. 명령행 도구 사용법
    print("\n6. 명령행 도구 사용법:")
    print("   # 설정 파일 생성")
    print("   s3-uploader-config --preset development --output config.json")
    print("   ")
    print("   # 파이프라인 실행")
    print("   s3-uploader --video-dir /path/to/videos --s3-bucket your-bucket --s3-prefix sequences")
    
    print("\n=== 예시 완료 ===")
    print("실제 사용을 위해서는 위의 경로와 설정을 실제 값으로 변경하세요.")

if __name__ == "__main__":
    main() 