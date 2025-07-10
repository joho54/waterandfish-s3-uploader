#!/usr/bin/env python3
"""
S3 Uploader 기본 사용법 예시
"""

import os
from pathlib import Path
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig

def basic_example():
    """기본 사용법 예시"""
    print("=== S3 Uploader 기본 사용법 ===")
    
    # 1. 설정 파일 생성 (선택사항)
    config = PipelineConfig()
    config.set('video.target_fps', 30)
    config.set('mediapipe.model_complexity', 1)
    config.set('s3.max_workers', 4)
    
    # 설정 저장
    config.save_config('my_config.json')
    print("설정 파일 생성: my_config.json")
    
    # 2. 파이프라인 초기화
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir="/path/to/your/videos",  # 실제 비디오 디렉토리로 변경
        s3_bucket="your-bucket-name",      # 실제 S3 버킷 이름으로 변경
        s3_prefix="pose-sequences",
        aws_region="us-east-1"
    )
    
    # 3. 파이프라인 실행
    result = pipeline.run_streaming_pipeline(
        target_fps=30,
        max_frames=None  # 전체 프레임 처리
    )
    
    # 4. 결과 출력
    print(f"\n처리 결과:")
    print(f"  - 총 비디오 수: {result['statistics']['total_videos']}")
    print(f"  - 성공: {result['statistics']['processed_videos']}개")
    print(f"  - 실패: {result['statistics']['failed_videos']}개")
    print(f"  - 실행 시간: {result['duration']:.2f}초")

def custom_config_example():
    """커스텀 설정 예시"""
    print("\n=== 커스텀 설정 예시 ===")
    
    # 고정확도 설정
    config = PipelineConfig()
    config.set('video.target_fps', 60)
    config.set('mediapipe.model_complexity', 2)
    config.set('mediapipe.min_detection_confidence', 0.8)
    config.set('mediapipe.min_tracking_confidence', 0.8)
    config.set('s3.max_workers', 2)
    
    # 설정 유효성 검사
    errors = config.validate_config()
    if errors:
        print("설정 오류:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # 파이프라인 초기화 (커스텀 설정 적용)
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir="/path/to/your/videos",
        s3_bucket="your-bucket-name",
        s3_prefix="high-quality-sequences",
        **config.get_extractor_kwargs()
    )
    
    # 파이프라인 실행
    result = pipeline.run_streaming_pipeline(
        **config.get_video_kwargs()
    )
    
    print(f"고정확도 처리 완료: {result['statistics']['processed_videos']}개")

def batch_processing_example():
    """배치 처리 예시"""
    print("\n=== 배치 처리 예시 ===")
    
    # 여러 비디오 디렉토리 처리
    video_dirs = [
        "/path/to/videos/class1",
        "/path/to/videos/class2", 
        "/path/to/videos/class3"
    ]
    
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):
            print(f"디렉토리가 존재하지 않습니다: {video_dir}")
            continue
            
        print(f"\n처리 중: {video_dir}")
        
        pipeline = MediaPipeS3StreamingPipeline(
            video_dir=video_dir,
            s3_bucket="your-bucket-name",
            s3_prefix=f"sequences/{Path(video_dir).name}"
        )
        
        result = pipeline.run_streaming_pipeline(
            target_fps=30,
            max_frames=None
        )
        
        print(f"  완료: {result['statistics']['processed_videos']}개")

def individual_components_example():
    """개별 컴포넌트 사용 예시"""
    print("\n=== 개별 컴포넌트 사용 예시 ===")
    
    from s3_uploader import MediaPipeStreamingExtractor, S3StreamingUploader
    
    # 1. 추출기만 사용
    extractor = MediaPipeStreamingExtractor(
        model_complexity=1,
        min_detection_confidence=0.5
    )
    
    video_path = "/path/to/single/video.mp4"
    if os.path.exists(video_path):
        print(f"시퀀스 추출 중: {video_path}")
        
        compressed_data, metadata = extractor.extract_sequence_to_memory(
            video_path=video_path,
            target_fps=30
        )
        
        print(f"  추출 완료: {metadata['extracted_frames']} 프레임")
        print(f"  파일 크기: {metadata['file_size_mb']:.2f} MB")
        
        # 2. 업로더만 사용
        uploader = S3StreamingUploader(
            bucket_name="your-bucket-name",
            region_name="us-east-1"
        )
        
        s3_key = f"sequences/{Path(video_path).stem}_landmarks.pkl.gz"
        
        upload_result = uploader.upload_data_streaming(
            data=compressed_data,
            s3_key=s3_key
        )
        
        print(f"  업로드 완료: {upload_result['status']}")
        if upload_result['status'] == 'success':
            print(f"  업로드 속도: {upload_result['speed_mbps']:.2f} MB/s")

def main():
    """메인 함수"""
    print("S3 Uploader 예시 실행")
    print("실제 사용하기 전에 경로와 버킷 이름을 수정하세요.\n")
    
    # AWS 자격 증명 확인
    if not os.environ.get('AWS_ACCESS_KEY_ID'):
        print("경고: AWS_ACCESS_KEY_ID 환경 변수가 설정되지 않았습니다.")
        print("AWS CLI를 사용하여 자격 증명을 설정하세요:")
        print("  aws configure")
        print()
    
    # 예시 실행
    basic_example()
    custom_config_example()
    batch_processing_example()
    individual_components_example()
    
    print("\n=== 예시 완료 ===")
    print("실제 사용 시 다음 사항을 확인하세요:")
    print("1. AWS 자격 증명 설정")
    print("2. S3 버킷 생성 및 권한 설정")
    print("3. 비디오 파일 경로 수정")
    print("4. 네트워크 연결 상태 확인")

if __name__ == "__main__":
    main() 