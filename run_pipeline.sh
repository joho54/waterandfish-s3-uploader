#!/bin/bash

# MediaPipe S3 Streaming Pipeline 실행 스크립트
# 사용법: ./run_pipeline.sh [옵션]

set -e  # 오류 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 기본 설정
DEFAULT_CONFIG="spec14.json"
DEFAULT_S3_BUCKET="waterandfish-s3"
DEFAULT_S3_PREFIX="feature-extraction-cache"
DEFAULT_REGION="ap-northeast-2"

# 도움말 함수
show_help() {
    echo "MediaPipe S3 Streaming Pipeline 실행 스크립트"
    echo ""
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -c, --config FILE     설정 파일 경로 (기본값: $DEFAULT_CONFIG)"
    echo "  -b, --bucket NAME     S3 버킷 이름 (기본값: $DEFAULT_S3_BUCKET)"
    echo "  -p, --prefix PREFIX   S3 업로드 경로 prefix (기본값: $DEFAULT_S3_PREFIX)"
    echo "  -r, --region REGION   AWS 리전 (기본값: $DEFAULT_REGION)"
    echo "  --no-upload          S3 업로드 비활성화"
    echo "  -h, --help           이 도움말 표시"
    echo ""
    echo "예시:"
    echo "  $0                                    # 기본 설정으로 실행"
    echo "  $0 -c spec14.json                     # 특정 설정 파일 사용"
    echo "  $0 --bucket my-bucket --no-upload     # S3 업로드 없이 실행"
}

# 명령행 인수 파싱
CONFIG_FILE="$DEFAULT_CONFIG"
S3_BUCKET="$DEFAULT_S3_BUCKET"
S3_PREFIX="$DEFAULT_S3_PREFIX"
REGION="$DEFAULT_REGION"
UPLOAD_FLAG="--upload"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b|--bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        -p|--prefix)
            S3_PREFIX="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        --no-upload)
            UPLOAD_FLAG=""
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# 메인 실행 함수
main() {
    log_info "MediaPipe S3 Streaming Pipeline 시작"
    log_info "설정 파일: $CONFIG_FILE"
    log_info "S3 버킷: $S3_BUCKET"
    log_info "S3 Prefix: $S3_PREFIX"
    log_info "AWS 리전: $REGION"
    log_info "S3 업로드: $([ -n "$UPLOAD_FLAG" ] && echo "활성화" || echo "비활성화")"
    
    # 1. Python 환경 확인
    log_info "Python 환경 확인 중..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python3가 설치되어 있지 않습니다."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_success "Python 버전: $PYTHON_VERSION"
    
    # 2. 필요한 파일들 확인
    log_info "필요한 파일들 확인 중..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
        exit 1
    fi
    
    if [[ ! -f "mediapipe_s3_streaming_pipeline_integrated.py" ]]; then
        log_error "메인 파이프라인 파일을 찾을 수 없습니다: mediapipe_s3_streaming_pipeline_integrated.py"
        exit 1
    fi
    
    if [[ ! -f "config.py" ]]; then
        log_error "설정 파일을 찾을 수 없습니다: config.py"
        exit 1
    fi
    
    if [[ ! -f "labels.csv" ]]; then
        log_error "라벨 파일을 찾을 수 없습니다: labels.csv"
        exit 1
    fi
    
    log_success "모든 필요한 파일이 존재합니다"
    
    # 3. Conda 환경 활성화
    log_info "Conda 환경 '/Users/johyeonho/SaturdayDinner/.conda' 활성화 중..."
    if command -v conda &> /dev/null; then
        if conda env list | grep -q "/Users/johyeonho/SaturdayDinner/.conda"; then
            eval "$(conda shell.bash hook)"
            conda activate "/Users/johyeonho/SaturdayDinner/.conda"
            log_success "Conda 환경이 활성화되었습니다"
        else
            log_error "Conda 환경 '/Users/johyeonho/SaturdayDinner/.conda'을 찾을 수 없습니다."
            log_info "사용 가능한 환경 목록:"
            conda env list
            exit 1
        fi
    else
        log_error "Conda가 설치되어 있지 않습니다."
        exit 1
    fi
    
    # 4. 의존성 설치 확인
    log_info "Python 패키지 의존성 확인 중..."
    if ! python3 -c "import mediapipe, boto3, numpy, pandas, tqdm, scipy" 2>/dev/null; then
        log_warning "일부 필요한 패키지가 설치되어 있지 않습니다."
        log_info "requirements.txt에서 패키지를 설치하시겠습니까? (y/N)"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            log_info "패키지 설치 중..."
            pip3 install -r requirements.txt
            log_success "패키지 설치 완료"
        else
            log_error "필요한 패키지가 설치되어 있지 않아 실행할 수 없습니다."
            exit 1
        fi
    else
        log_success "모든 필요한 패키지가 설치되어 있습니다"
    fi
    
    # 5. AWS 자격 증명 확인 (S3 업로드가 활성화된 경우)
    if [[ -n "$UPLOAD_FLAG" ]]; then
        log_info "AWS 자격 증명 확인 중..."
        if ! aws sts get-caller-identity &>/dev/null; then
            log_warning "AWS 자격 증명이 설정되어 있지 않습니다."
            log_info "AWS CLI를 사용하여 자격 증명을 설정하거나 환경 변수를 설정해주세요."
            log_info "예시:"
            log_info "  export AWS_ACCESS_KEY_ID=your_access_key"
            log_info "  export AWS_SECRET_ACCESS_KEY=your_secret_key"
            log_info "  export AWS_DEFAULT_REGION=$REGION"
            log_info ""
            log_info "계속 진행하시겠습니까? (y/N)"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
            log_success "AWS 자격 증명이 설정되어 있습니다"
        fi
    fi
    
    # 6. 파이프라인 실행
    log_info "MediaPipe S3 Streaming Pipeline 실행 중..."
    log_info "이 과정은 시간이 오래 걸릴 수 있습니다..."
    
    # 실행 명령어 구성
    CMD="python3 mediapipe_s3_streaming_pipeline_integrated.py"
    CMD="$CMD --config $CONFIG_FILE"
    CMD="$CMD --s3-bucket $S3_BUCKET"
    CMD="$CMD --s3-prefix $S3_PREFIX"
    CMD="$CMD --region $REGION"
    if [[ -n "$UPLOAD_FLAG" ]]; then
        CMD="$CMD $UPLOAD_FLAG"
    fi
    
    log_info "실행 명령어: $CMD"
    echo ""
    
    # 파이프라인 실행
    if eval "$CMD"; then
        log_success "MediaPipe S3 Streaming Pipeline이 성공적으로 완료되었습니다!"
    else
        log_error "파이프라인 실행 중 오류가 발생했습니다."
        exit 1
    fi
}

# 스크립트 실행
main "$@"
