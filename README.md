# S3 Uploader Package

미디어파이프 시퀀스 추출 및 S3 스트리밍 업로드 파이프라인 패키지

## 📦 설치

### PyPI에서 설치 (권장)
```bash
pip install s3-uploader
```

### 소스에서 설치
```bash
git clone https://github.com/your-username/s3-uploader.git
cd s3-uploader
pip install -e .
```

## 🚀 빠른 시작

### 1. AWS 설정
```bash
# AWS CLI 설치
pip install awscli

# AWS 자격 증명 설정
aws configure
```

### 2. 기본 사용법
```bash
# 설정 파일 생성
s3-uploader-config --preset development --output config.json

# 파이프라인 실행
s3-uploader \
  --video-dir /path/to/videos \
  --s3-bucket your-bucket-name \
  --s3-prefix sequences
```

### 3. Python 코드에서 사용
```python
from s3_uploader import MediaPipeS3StreamingPipeline

# 파이프라인 초기화
pipeline = MediaPipeS3StreamingPipeline(
    video_dir="/path/to/videos",
    s3_bucket="your-bucket-name",
    s3_prefix="sequences",
    aws_region="us-east-1"
)

# 파이프라인 실행
result = pipeline.run_streaming_pipeline(
    target_fps=30,
    max_frames=None
)

print(f"처리 완료: {result['processed_videos']}개 비디오")
```

## 📋 주요 기능

- **미디어파이프 시퀀스 추출**: 포즈, 손 랜드마크 추출
- **메모리 스트리밍**: 로컬 저장 없이 직접 S3 업로드
- **병렬 처리**: 멀티스레드 업로드 지원
- **압축 최적화**: pickle + gzip 압축으로 크기 절약
- **설정 관리**: JSON 기반 설정 파일 지원

## ⚙️ 설정

### 환경별 프리셋
```bash
# 개발 환경 (빠른 테스트)
s3-uploader-config --preset development

# 프로덕션 환경 (고품질)
s3-uploader-config --preset production

# 고정확도 환경 (연구용)
s3-uploader-config --preset high_accuracy

# 빠른 처리 환경
s3-uploader-config --preset fast_processing
```

### 커스텀 설정
```json
{
  "video": {
    "target_fps": 30,
    "max_frames": null
  },
  "mediapipe": {
    "model_complexity": 1,
    "min_detection_confidence": 0.5
  },
  "s3": {
    "max_workers": 4,
    "chunk_size": 8388608
  }
}
```

## 🔧 고급 사용법

### 개별 컴포넌트 사용
```python
from s3_uploader import MediaPipeStreamingExtractor, S3StreamingUploader

# 추출기만 사용
extractor = MediaPipeStreamingExtractor(
    model_complexity=1,
    min_detection_confidence=0.5
)

compressed_data, metadata = extractor.extract_sequence_to_memory(
    video_path="video.mp4",
    target_fps=30
)

# 업로더만 사용
uploader = S3StreamingUploader(
    bucket_name="your-bucket",
    region_name="us-east-1"
)

result = uploader.upload_data_streaming(
    data=compressed_data,
    s3_key="sequences/video_landmarks.pkl.gz"
)
```

### 배치 처리
```python
import glob
from pathlib import Path

# 여러 비디오 디렉토리 처리
video_dirs = glob.glob("/path/to/videos/*")

for video_dir in video_dirs:
    pipeline = MediaPipeS3StreamingPipeline(
        video_dir=video_dir,
        s3_bucket="your-bucket",
        s3_prefix=f"sequences/{Path(video_dir).name}"
    )
    
    result = pipeline.run_streaming_pipeline()
    print(f"{video_dir}: {result['processed_videos']}개 처리")
```

## 📊 성능 최적화

### 권장 설정

| 환경 | target_fps | model_complexity | max_workers | 메모리 사용량 |
|------|------------|------------------|-------------|---------------|
| 개발 | 15 | 0 | 2 | 4GB |
| 프로덕션 | 30 | 1 | 4 | 8GB |
| 고정확도 | 60 | 2 | 2 | 16GB |

### 성능 팁

1. **GPU 사용**: CUDA 지원 GPU가 있으면 처리 속도 향상
2. **병렬 처리**: `max_workers` 증가로 업로드 속도 향상
3. **메모리 관리**: `max_frames` 설정으로 메모리 사용량 제어
4. **네트워크 최적화**: AWS 리전과 가까운 위치에서 실행

## 🐛 문제 해결

### 일반적인 문제

#### 1. 미디어파이프 설치 오류
```bash
pip uninstall opencv-python
pip install opencv-python-headless
pip install mediapipe
```

#### 2. AWS 권한 오류
```bash
aws sts get-caller-identity
aws s3 ls s3://your-bucket-name
```

#### 3. 메모리 부족
```bash
# 설정에서 메모리 제한 조정
{
  "performance": {
    "memory_limit_gb": 4
  }
}
```

## 📈 예시

### 기본 예시
```python
from s3_uploader import MediaPipeS3StreamingPipeline

pipeline = MediaPipeS3StreamingPipeline(
    video_dir="/Volumes/ExternalHD/videos",
    s3_bucket="my-ml-bucket",
    s3_prefix="pose-sequences"
)

result = pipeline.run_streaming_pipeline(
    target_fps=30,
    max_frames=None
)

print(f"성공: {result['processed_videos']}개")
print(f"실패: {result['failed_videos']}개")
```

### 고급 예시
```python
from s3_uploader import MediaPipeS3StreamingPipeline
from s3_uploader import PipelineConfig

# 커스텀 설정
config = PipelineConfig()
config.set('video.target_fps', 60)
config.set('mediapipe.model_complexity', 2)
config.set('s3.max_workers', 8)

pipeline = MediaPipeS3StreamingPipeline(
    video_dir="/path/to/videos",
    s3_bucket="your-bucket",
    s3_prefix="high-quality-sequences",
    **config.get_extractor_kwargs()
)

result = pipeline.run_streaming_pipeline()
```

## 📚 API 문서

### MediaPipeS3StreamingPipeline

메인 파이프라인 클래스

```python
class MediaPipeS3StreamingPipeline:
    def __init__(
        self,
        video_dir: str,
        s3_bucket: str,
        s3_prefix: str,
        aws_region: str = "us-east-1",
        **kwargs
    ):
        """
        Args:
            video_dir: 비디오 파일이 있는 디렉토리 경로
            s3_bucket: S3 버킷 이름
            s3_prefix: S3 키 접두사
            aws_region: AWS 리전
            **kwargs: MediaPipe 설정 파라미터
        """
    
    def run_streaming_pipeline(
        self,
        target_fps: int = 30,
        max_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        스트리밍 파이프라인 실행
        
        Returns:
            Dict containing processing results
        """
```

### MediaPipeStreamingExtractor

미디어파이프 추출기

```python
class MediaPipeStreamingExtractor:
    def extract_sequence_to_memory(
        self,
        video_path: str,
        target_fps: int = 30
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        비디오에서 랜드마크 시퀀스 추출
        
        Returns:
            Tuple of (compressed_data, metadata)
        """
```

### S3StreamingUploader

S3 업로더

```python
class S3StreamingUploader:
    def upload_data_streaming(
        self,
        data: bytes,
        s3_key: str
    ) -> Dict[str, Any]:
        """
        데이터를 S3에 스트리밍 업로드
        
        Returns:
            Dict containing upload results
        """
```

## 🔗 의존성

- **mediapipe**: 포즈 랜드마크 추출
- **opencv-python**: 비디오 처리
- **boto3**: AWS S3 업로드
- **numpy**: 수치 계산
- **pickle**: 데이터 직렬화
- **gzip**: 데이터 압축

## 📄 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요. 