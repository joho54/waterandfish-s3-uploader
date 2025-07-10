# 미디어파이프 S3 파이프라인

로컬 비디오에서 미디어파이프 포즈 시퀀스를 추출하고 S3에 업로드하는 통합 파이프라인입니다.

## 📋 목차

1. [개요](#개요)
2. [주요 기능](#주요-기능)
3. [설치 및 설정](#설치-및-설정)
4. [사용법](#사용법)
5. [파이프라인 구성](#파이프라인-구성)
6. [설정 옵션](#설정-옵션)
7. [성능 최적화](#성능-최적화)
8. [문제 해결](#문제-해결)
9. [예시](#예시)

## 🎯 개요

이 파이프라인은 다음과 같은 작업을 수행합니다:

1. **비디오 시퀀스 추출**: 로컬 비디오 파일에서 미디어파이프를 사용하여 포즈 랜드마크 시퀀스 추출
2. **데이터 압축**: 추출된 시퀀스를 pickle + gzip으로 압축하여 저장 공간 절약
3. **S3 업로드**: 압축된 시퀀스를 AWS S3에 병렬 업로드
4. **매니페스트 생성**: 처리 결과를 JSON 매니페스트로 기록

### 장점

- **저장 공간 절약**: 원본 비디오 대비 90% 이상 크기 감소
- **네트워크 효율성**: 압축된 시퀀스만 업로드하여 대역폭 절약
- **처리 속도**: 미디어파이프의 빠른 포즈 검출
- **확장성**: 배치 처리 및 병렬 업로드 지원

## ✨ 주요 기능

### 🔍 미디어파이프 시퀀스 추출
- **33개 포즈 랜드마크**: MediaPipe Pose의 모든 랜드마크 추출
- **FPS 조절**: 목표 FPS에 맞춰 다운샘플링
- **신뢰도 필터링**: 낮은 신뢰도의 프레임 자동 제외
- **배치 처리**: 여러 비디오 동시 처리

### ☁️ S3 업로드
- **병렬 업로드**: 멀티스레드로 동시 업로드
- **멀티파트 업로드**: 대용량 파일 자동 멀티파트 처리
- **중복 방지**: 파일 해시 기반 중복 업로드 방지
- **진행률 표시**: 실시간 업로드 진행률 모니터링

### ⚙️ 설정 관리
- **프리셋 설정**: 개발/프로덕션/고정확도/빠른처리 프리셋
- **유연한 설정**: JSON 기반 설정 파일
- **환경별 최적화**: 사용 사례에 맞는 설정 자동 적용

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
# 필수 패키지 설치
pip install -r pipeline_requirements.txt

# 또는 개별 설치
pip install mediapipe opencv-python boto3 numpy tqdm pandas
```

### 2. AWS 설정

```bash
# AWS CLI 설치 (macOS)
brew install awscli

# AWS 자격 증명 설정
aws configure

# 또는 환경 변수 설정
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 3. S3 버킷 생성

```bash
# S3 버킷 생성
aws s3 mb s3://your-unique-bucket-name

# 버킷 접근 확인
aws s3 ls s3://your-unique-bucket-name
```

## 📖 사용법

### 기본 사용법

```bash
# 쉘 스크립트 사용 (권장)
./run_pipeline.sh \
  -v /path/to/videos \
  -o ./output \
  -b your-bucket-name \
  -p sequences

./run_pipeline.sh \
  -v /path/to/videos \
  -o ./output \
  -b your-bucket-name \
  -p sequences

# Python 직접 실행
python mediapipe_s3_pipeline.py \
  --video-dir /path/to/videos \
  --output-dir ./output \
  --s3-bucket your-bucket-name \
  --s3-prefix sequences
```

### 고급 사용법

```bash
# 프로덕션 설정으로 실행
./run_pipeline.sh \
  -v /path/to/videos \
  -o ./output \
  -b your-bucket-name \
  -p sequences \
  --preset production \
  --fps 30

# 추출만 실행 (업로드 건너뛰기)
./run_pipeline.sh \
  -v /path/to/videos \
  -o ./output \
  -b your-bucket-name \
  -p sequences \
  --skip-upload

# 드라이 런 (실제 실행 없이 명령어만 출력)
./run_pipeline.sh \
  -v /path/to/videos \
  -o ./output \
  -b your-bucket-name \
  -p sequences \
  --dry-run
```

### 개별 컴포넌트 사용

```bash
# 시퀀스 추출만
python mediapipe_sequence_extractor.py \
  --input /path/to/videos \
  --output ./sequences \
  --batch \
  --target-fps 30

# S3 업로드만
python s3_uploader.py \
  --local-dir ./sequences \
  --bucket your-bucket-name \
  --s3-prefix sequences
```

## 🏗️ 파이프라인 구성

### 파일 구조

```
mobilenet-finetuning/
├── mediapipe_sequence_extractor.py    # 시퀀스 추출기
├── s3_uploader.py                     # S3 업로더
├── mediapipe_s3_pipeline.py           # 통합 파이프라인
├── pipeline_config.py                 # 설정 관리
├── run_pipeline.sh                    # 실행 스크립트
├── pipeline_requirements.txt          # 의존성 패키지
└── MEDIAPIPE_S3_PIPELINE_README.md   # 이 파일
```

### 파이프라인 흐름

```
1. 비디오 파일 스캔
   ↓
2. 미디어파이프 포즈 추출
   ↓
3. 시퀀스 압축 (pickle + gzip)
   ↓
4. 로컬 저장
   ↓
5. S3 병렬 업로드
   ↓
6. 매니페스트 생성
```

## ⚙️ 설정 옵션

### 설정 파일 생성

```bash
# 개발 환경 설정
python pipeline_config.py --preset development --output dev_config.json

# 프로덕션 환경 설정
python pipeline_config.py --preset production --output prod_config.json

# 고정확도 설정
python pipeline_config.py --preset high_accuracy --output accuracy_config.json
```

### 주요 설정 옵션

#### 비디오 처리 설정
```json
{
  "video": {
    "target_fps": 30,           // 목표 FPS
    "max_frames": null,         // 최대 프레임 수 (null = 전체)
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv"]
  }
}
```

#### 미디어파이프 설정
```json
{
  "mediapipe": {
    "model_complexity": 1,      // 0=빠름, 1=균형, 2=정확
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
  }
}
```

#### S3 설정
```json
{
  "s3": {
    "max_workers": 4,           // 동시 업로드 스레드 수
    "chunk_size": 8388608,      // 멀티파트 청크 크기 (8MB)
    "overwrite": false
  }
}
```

## 🚀 성능 최적화

### 환경별 권장 설정

#### 개발 환경 (빠른 테스트)
```bash
./run_pipeline.sh \
  --preset development \
  --fps 15 \
  --max-frames 300
```

#### 프로덕션 환경 (고품질)
```bash
./run_pipeline.sh \
  --preset production \
  --fps 30
```

#### 고정확도 환경 (연구용)
```bash
./run_pipeline.sh \
  --preset high_accuracy \
  --fps 60
```

### 성능 팁

1. **GPU 가속**: CUDA 지원 GPU 사용 시 처리 속도 향상
2. **병렬 처리**: `s3.max_workers` 증가로 업로드 속도 향상
3. **메모리 최적화**: `performance.memory_limit_gb` 설정으로 메모리 사용량 제어
4. **네트워크 최적화**: AWS 리전과 가까운 위치에서 실행

## 🔧 문제 해결

### 일반적인 문제

#### 1. 미디어파이프 설치 오류
```bash
# OpenCV 재설치
pip uninstall opencv-python
pip install opencv-python-headless

# 미디어파이프 재설치
pip uninstall mediapipe
pip install mediapipe
```

#### 2. AWS 권한 오류
```bash
# IAM 정책 확인
aws iam get-user
aws sts get-caller-identity

# S3 권한 테스트
aws s3 ls s3://your-bucket-name
```

#### 3. 메모리 부족 오류
```bash
# 배치 크기 줄이기
python pipeline_config.py --preset development

# 또는 수동 설정
{
  "performance": {
    "memory_limit_gb": 4,
    "batch_size": 1
  }
}
```

#### 4. 네트워크 타임아웃
```bash
# 업로드 워커 수 줄이기
{
  "s3": {
    "max_workers": 2
  }
}
```

### 디버깅 모드

```bash
# 상세 로그 출력
export PYTHONPATH=.
python -u mediapipe_s3_pipeline.py \
  --video-dir /path/to/videos \
  --output-dir ./output \
  --s3-bucket your-bucket \
  --s3-prefix sequences \
  2>&1 | tee pipeline.log
```

## 📊 예시

### 예시 1: 기본 실행

```bash
# 1. 설정 파일 생성
python pipeline_config.py --preset development --output config.json

# 2. 파이프라인 실행
./run_pipeline.sh \
  -v /Volumes/ExternalHD/videos \
  -o ./sequences \
  -b my-ml-bucket \
  -p pose-sequences \
  -c config.json

# 3. 결과 확인
ls -la ./sequences/
aws s3 ls s3://my-ml-bucket/pose-sequences/
```

### 예시 2: 대용량 데이터 처리

```bash
# 1. 프로덕션 설정
python pipeline_config.py --preset production --output prod_config.json

# 2. 배치 처리 (여러 디렉토리)
for dir in /Volumes/ExternalHD/videos/*; do
  ./run_pipeline.sh \
    -v "$dir" \
    -o "./sequences/$(basename "$dir")" \
    -b my-ml-bucket \
    -p "sequences/$(basename "$dir")" \
    -c prod_config.json
done
```

### 예시 3: 단계별 실행

```bash
# 1단계: 시퀀스 추출만
python mediapipe_sequence_extractor.py \
  --input /Volumes/ExternalHD/videos \
  --output ./sequences \
  --batch \
  --target-fps 30

# 2단계: S3 업로드만
python s3_uploader.py \
  --local-dir ./sequences \
  --bucket my-ml-bucket \
  --s3-prefix sequences \
  --max-workers 8
```

## 📈 성능 벤치마크

### 처리 속도 (예시)

| 설정 | 비디오 길이 | 처리 시간 | 압축률 | 파일 크기 |
|------|-------------|-----------|--------|-----------|
| 개발 (15fps, 300프레임) | 10초 | 30초 | 95% | 50KB |
| 프로덕션 (30fps, 전체) | 60초 | 2분 | 92% | 200KB |
| 고정확도 (60fps, 전체) | 60초 | 4분 | 90% | 400KB |

### 네트워크 효율성

- **원본 비디오**: 100MB → S3 업로드 시간: 10분
- **압축 시퀀스**: 200KB → S3 업로드 시간: 10초
- **효율성 향상**: 60배 빠른 업로드

## 🤝 기여

버그 리포트나 기능 요청은 이슈를 통해 제출해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 