#!/usr/bin/env python3
"""
파이프라인 디버깅을 위한 테스트 스크립트
"""

import os
import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_video(video_path: str):
    """단일 비디오 테스트"""
    from s3_uploader.mediapipe_s3_streaming_pipeline import MediaPipeS3StreamingPipeline
    
    # 테스트용 설정
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir=str(Path(video_path).parent),
        s3_bucket="waterandfish-s3",
        s3_prefix="test-sign-language-data",
        aws_region="us-east-1",
        max_workers=1,  # 단일 스레드로 테스트
        model_complexity=1,  # 빠른 테스트를 위해 복잡도 낮춤
        min_detection_confidence=0.3,  # 더 관대한 설정
        min_tracking_confidence=0.3
    )
    
    # 단일 비디오 처리
    result = pipeline.process_video_streaming(
        Path(video_path),
        target_fps=15,  # 더 낮은 FPS로 테스트
        max_frames=100  # 최대 100프레임만 테스트
    )
    
    logger.info(f"테스트 결과: {result}")
    return result

def test_video_directory(video_dir: str, max_videos: int = 5):
    """비디오 디렉토리 테스트"""
    from s3_uploader.mediapipe_s3_streaming_pipeline import MediaPipeS3StreamingPipeline
    
    # 테스트용 설정
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir=video_dir,
        s3_bucket="waterandfish-s3",
        s3_prefix="test-sign-language-data",
        aws_region="us-east-1",
        max_workers=2,  # 적은 스레드로 테스트
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    # 비디오 파일 찾기
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    # 최대 개수 제한
    video_files = video_files[:max_videos]
    
    logger.info(f"테스트할 비디오 파일: {len(video_files)}개")
    
    results = []
    for video_file in video_files:
        logger.info(f"테스트 중: {video_file.name}")
        try:
            result = pipeline.process_video_streaming(
                video_file,
                target_fps=15,
                max_frames=50  # 더 적은 프레임으로 테스트
            )
            results.append(result)
        except Exception as e:
            logger.error(f"테스트 실패 {video_file}: {e}")
            results.append({
                'video_path': str(video_file),
                'error': str(e),
                'status': 'failed'
            })
    
    # 결과 분석
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"테스트 완료: 성공 {len(successful)}개, 실패 {len(failed)}개")
    
    if failed:
        logger.info("실패한 파일들:")
        for f in failed:
            logger.info(f"  {Path(f['video_path']).name}: {f.get('error', 'unknown')}")
    
    return results

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['--help', '-h']:
        print("사용법:")
        print("  python test_pipeline_debug.py <video_path>")
        print("  python test_pipeline_debug.py <video_dir> [max_videos]")
        print("")
        print("예시:")
        print("  python test_pipeline_debug.py /path/to/video.mp4")
        print("  python test_pipeline_debug.py /path/to/video/directory 10")
        sys.exit(1)
    
    path = sys.argv[1]
    max_videos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    if os.path.isfile(path):
        # 단일 파일 테스트
        test_single_video(path)
    elif os.path.isdir(path):
        # 디렉토리 테스트
        test_video_directory(path, max_videos)
    else:
        logger.error(f"경로가 존재하지 않습니다: {path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 