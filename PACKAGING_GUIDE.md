# S3 Uploader 패키징 가이드

이 문서는 S3 Uploader 패키지를 빌드하고 배포하는 방법을 설명합니다.

## 📦 패키지 구조

```
s3-uploader/
├── setup.py                 # 패키지 설정
├── pyproject.toml          # 현대적인 패키징 설정
├── requirements.txt        # 의존성 목록
├── README.md              # 패키지 README
├── LICENSE                # MIT 라이선스
├── MANIFEST.in            # 포함할 파일 목록
├── __init__.py            # 패키지 초기화
├── build_package.py       # 빌드 스크립트
├── mediapipe_s3_streaming_pipeline.py  # 메인 파이프라인
├── pipeline_config.py     # 설정 관리
├── MEDIAPIPE_S3_PIPELINE_README.md     # 상세 문서
├── PACKAGING_GUIDE.md     # 이 파일
└── examples/              # 사용 예시
    ├── __init__.py
    └── basic_usage.py
```

## 🚀 빠른 시작

### 1. 패키지 빌드

```bash
# 빌드 스크립트 사용 (권장)
python build_package.py build

# 또는 직접 빌드
python -m build --sdist --wheel
```

### 2. 테스트 설치

```bash
# 빌드된 패키지 테스트 설치
python build_package.py test
```

### 3. TestPyPI 업로드

```bash
# TestPyPI에 업로드 (테스트용)
python build_package.py testpypi
```

### 4. PyPI 업로드

```bash
# PyPI에 업로드 (실제 배포)
python build_package.py pypi
```

## 📋 빌드 전 체크리스트

### 필수 파일 확인

- [ ] `setup.py` 또는 `pyproject.toml` 존재
- [ ] `README.md` 작성 완료
- [ ] `LICENSE` 파일 포함
- [ ] `requirements.txt` 의존성 정리
- [ ] `__init__.py`에서 주요 클래스 export
- [ ] `MANIFEST.in` 파일 포함 목록 정의

### 코드 품질 확인

- [ ] 모든 import 문 정상 작동
- [ ] 함수/클래스 docstring 작성
- [ ] 예외 처리 구현
- [ ] 로깅 설정 완료
- [ ] 설정 파일 검증 로직 구현

### 테스트 확인

- [ ] 기본 기능 동작 테스트
- [ ] 설정 파일 로드 테스트
- [ ] AWS 연결 테스트
- [ ] 메모리 사용량 확인

## 🔧 빌드 스크립트 사용법

### 전체 과정

```bash
# 모든 과정을 한 번에 실행
python build_package.py all
```

### 단계별 실행

```bash
# 1. 빌드 디렉토리 정리
python build_package.py clean

# 2. 패키지 빌드
python build_package.py build

# 3. 테스트 설치
python build_package.py test

# 4. TestPyPI 업로드
python build_package.py testpypi

# 5. PyPI 업로드
python build_package.py pypi
```

## 📦 배포 전 준비사항

### 1. PyPI 계정 설정

```bash
# PyPI 계정 생성 (https://pypi.org/account/register/)
# TestPyPI 계정 생성 (https://test.pypi.org/account/register/)

# twine 설정
pip install twine

# ~/.pypirc 파일 생성
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = your-username
password = your-password

[testpypi]
repository = https://test.pypi.org/legacy/
username = your-username
password = your-password
```

### 2. 패키지 이름 확인

- PyPI에서 사용 가능한 이름인지 확인
- 다른 패키지와 충돌하지 않는지 확인
- 이름이 명확하고 기억하기 쉬운지 확인

### 3. 버전 관리

```python
# __init__.py에서 버전 업데이트
__version__ = "1.0.1"  # 새로운 버전

# setup.py에서 버전 업데이트
version="1.0.1"
```

## 🧪 테스트 배포 (TestPyPI)

### 1. TestPyPI 업로드

```bash
python build_package.py testpypi
```

### 2. TestPyPI에서 설치 테스트

```bash
# TestPyPI에서 설치
pip install --index-url https://test.pypi.org/simple/ s3-uploader

# 설치 확인
python -c "import s3_uploader; print(s3_uploader.__version__)"
```

### 3. 기능 테스트

```bash
# 기본 기능 테스트
python -c "
from s3_uploader import MediaPipeS3StreamingPipeline, PipelineConfig
print('패키지 import 성공')
"
```

## 🚀 실제 배포 (PyPI)

### 1. 최종 확인

- [ ] 모든 테스트 통과
- [ ] 문서 완성
- [ ] 예시 코드 동작 확인
- [ ] 라이선스 확인

### 2. PyPI 업로드

```bash
python build_package.py pypi
```

### 3. 배포 확인

```bash
# PyPI에서 설치
pip install s3-uploader

# 설치 확인
python -c "import s3_uploader; print(s3_uploader.__version__)"
```

## 🔄 버전 업데이트

### 1. 버전 번호 업데이트

```python
# __init__.py
__version__ = "1.0.1"

# setup.py
version="1.0.1"

# pyproject.toml (동적 버전 사용 시)
# setuptools_scm이 자동으로 관리
```

### 2. 변경사항 기록

```markdown
# CHANGELOG.md
## [1.0.1] - 2024-01-01
### Added
- 새로운 기능 추가

### Changed
- 기존 기능 개선

### Fixed
- 버그 수정
```

### 3. 재배포

```bash
# 새 버전 빌드 및 배포
python build_package.py all
python build_package.py pypi
```

## 🐛 문제 해결

### 일반적인 문제

#### 1. 빌드 실패

```bash
# 의존성 확인
pip install --upgrade setuptools wheel build

# 캐시 정리
python build_package.py clean
```

#### 2. 업로드 실패

```bash
# 자격 증명 확인
cat ~/.pypirc

# 패키지 검사
twine check dist/*
```

#### 3. 설치 실패

```bash
# 가상환경에서 테스트
python -m venv test_env
source test_env/bin/activate  # Windows: test_env\Scripts\activate
pip install s3-uploader
```

### 디버깅 팁

1. **로컬 테스트**: `pip install -e .`로 개발 모드 설치
2. **의존성 확인**: `pip check`로 의존성 충돌 확인
3. **패키지 검사**: `twine check`로 배포 전 검사
4. **가상환경 사용**: 깨끗한 환경에서 테스트

## 📚 추가 자료

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Upload Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [twine Documentation](https://twine.readthedocs.io/)

## 🤝 기여

패키징 개선 제안이나 버그 리포트는 GitHub Issues를 통해 제출해주세요. 