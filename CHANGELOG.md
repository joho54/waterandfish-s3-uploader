# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup
- MediaPipe S3 Streaming Pipeline
- Configuration management system
- Build and packaging scripts
- Comprehensive documentation

### Changed
- N/A

### Fixed
- N/A

## [1.0.0] - 2024-01-01

### Added
- MediaPipeStreamingExtractor: MediaPipe를 사용한 실시간 포즈 추출
- S3StreamingUploader: AWS S3에 실시간 스트리밍 업로드
- MediaPipeS3StreamingPipeline: 통합 파이프라인 클래스
- PipelineConfig: 설정 관리 클래스
- ConfigPresets: 미리 정의된 설정 프리셋
- create_config_file: 설정 파일 생성 유틸리티
- build_package.py: 자동화된 빌드 및 배포 스크립트
- PACKAGING_GUIDE.md: 상세한 패키징 가이드
- MEDIAPIPE_S3_PIPELINE_README.md: 파이프라인 사용법 가이드
- example_config.json: 설정 파일 예시
- usage_example.py: 사용 예시 코드
- run_sign_language_pipeline.sh: 실행 스크립트

### Features
- 실시간 비디오 스트림에서 포즈 추출
- AWS S3에 실시간 업로드
- 설정 기반 파이프라인 구성
- 메모리 효율적인 스트리밍 처리
- 로깅 및 에러 처리
- 가상환경 지원

### Technical Details
- Python 3.8+ 지원
- MediaPipe 0.10.0+ 의존성
- boto3 AWS SDK 통합
- OpenCV 비디오 처리
- JSON 기반 설정 시스템 