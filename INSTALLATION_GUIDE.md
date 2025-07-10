# S3 Uploader 설치 및 사용 가이드

이 문서는 다른 PC에서 S3 Uploader 프로젝트를 설치하고 사용하는 방법을 설명합니다.

## 📋 시스템 요구사항

### 필수 요구사항
- Python 3.8 이상
- pip (Python 패키지 관리자)
- Git

### 권장사항
- Conda 또는 Miniconda (가상환경 관리)
- 8GB 이상 RAM (MediaPipe 처리용)
- 웹캠 또는 비디오 파일

## 🚀 빠른 설치

### 1. 저장소 클론

```bash
# GitHub에서 클론 (GitHub에 업로드 후)
git clone https://github.com/your-username/s3-uploader.git
cd s3-uploader

# 또는 로컬에서 복사한 경우
# 프로젝트 폴더로 이동
cd s3-uploader
```

### 2. 가상환경 생성 및 활성화

```bash
# Conda 사용 (권장)
conda create -n s3-uploader python=3.8
conda activate s3-uploader

# 또는 Python venv 사용
python -m venv s3-uploader-env
source s3-uploader-env/bin/activate  # macOS/Linux
# s3-uploader-env\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
# requirements.txt에서 설치
pip install -r requirements.txt

# 또는 개발 모드로 설치 (소스 코드 수정 시)
pip install -e .
```

## 📦 패키지 설치 (빌드된 패키지 사용)

### 1. 빌드된 패키지 설치

```bash
# 휠 파일 설치
pip install dist/s3_uploader-1.0.0-py3-none-any.whl

# 또는 소스 배포판 설치
pip install dist/s3_uploader-1.0.0.tar.gz
```

### 2. 설치 확인

```bash
python -c "import s3_uploader; print(s3_uploader.__version__)"
```

## 🔧 설정

### 1. AWS 자격 증명 설정

```bash
# AWS CLI 설치 (권장)
pip install awscli

# AWS 자격 증명 설정
aws configure

# 또는 환경 변수 설정
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. 설정 파일 생성

```bash
# 기본 설정 파일 생성
python -c "from s3_uploader import create_config_file; create_config_file('my_config.json')"

# 또는 example_config.json을 복사하여 수정
cp example_config.json my_config.json
```

### 3. 설정 파일 편집

`my_config.json` 파일을 편집하여 다음을 설정:

```json
{
  "aws": {
    "bucket_name": "your-s3-bucket",
    "region": "us-east-1"
  },
  "mediapipe": {
    "model_complexity": 1,
    "smooth_landmarks": true
  },
  "streaming": {
    "batch_size": 10,
    "upload_interval": 5.0
  }
}
```

## 🎯 사용 예시

### 1. 기본 사용법

```python
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig

# 설정 로드
config = PipelineConfig.from_file('my_config.json')

# 파이프라인 생성 및 실행
pipeline = MediaPipeS3StreamingPipeline(config)
pipeline.run()
```

### 2. 웹캠에서 실시간 처리

```python
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig

# 설정
config = PipelineConfig.from_file('my_config.json')
config.video_source = 0  # 웹캠

# 실행
pipeline = MediaPipeS3StreamingPipeline(config)
pipeline.run()
```

### 3. 비디오 파일 처리

```python
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig

# 설정
config = PipelineConfig.from_file('my_config.json')
config.video_source = "path/to/video.mp4"

# 실행
pipeline = MediaPipeS3StreamingPipeline(config)
pipeline.run()
```

## 🛠️ 문제 해결

### 일반적인 문제

#### 1. MediaPipe 설치 실패

```bash
# MediaPipe 재설치
pip uninstall mediapipe
pip install mediapipe==0.10.21

# 또는 conda 사용
conda install -c conda-forge mediapipe
```

#### 2. OpenCV 설치 실패

```bash
# OpenCV 재설치
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.11.0.86
```

#### 3. AWS 연결 실패

```bash
# AWS 자격 증명 확인
aws sts get-caller-identity

# 또는 boto3로 테스트
python -c "import boto3; print(boto3.client('s3').list_buckets())"
```

#### 4. 메모리 부족

```bash
# 설정에서 배치 크기 줄이기
# my_config.json에서 "batch_size"를 5로 줄이기
```

### 디버깅

#### 1. 로그 레벨 설정

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 테스트 실행

```bash
# 기본 테스트
python test_pipeline_debug.py

# 사용 예시 실행
python usage_example.py
```

## 📚 추가 자료

- [PACKAGING_GUIDE.md](PACKAGING_GUIDE.md) - 패키징 및 배포 가이드
- [MEDIAPIPE_S3_PIPELINE_README.md](MEDIAPIPE_S3_PIPELINE_README.md) - 상세한 파이프라인 설명
- [README.md](README.md) - 프로젝트 개요

## 🤝 지원

문제가 발생하면 다음을 확인하세요:

1. Python 버전이 3.8 이상인지 확인
2. 모든 의존성이 올바르게 설치되었는지 확인
3. AWS 자격 증명이 올바르게 설정되었는지 확인
4. 설정 파일이 올바른 형식인지 확인

추가 도움이 필요하면 GitHub Issues를 통해 문의해주세요. 