#!/bin/bash

# 수어 인식용 미디어파이프 S3 스트리밍 파이프라인 실행 스크립트
# Usage: ./run_sign_language_pipeline.sh

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

# 스크립트 시작
log_info "수어 인식 파이프라인 시작"

# 모든 비디오 디렉토리 정의
VIDEO_DIRS=(
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/0001~3000(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/3001~6000(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/6001~8280(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/8381~9000(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/9001~9600(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/9601~10480(영상)"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/10481~12994"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/12995~15508"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/15509~18022"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/18023~20536"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/20537~23050"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/23051~25564"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/25565~28078"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/28079~30592"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/30593~33106"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/33107~35620"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/36878~40027"
    "/Volumes/Sub_Storage/수어 데이터셋/수어 데이터셋/40028~43177"
)

# 전체 디렉토리 수
TOTAL_DIRS=${#VIDEO_DIRS[@]}
CURRENT_DIR=0
SUCCESS_COUNT=0
FAILED_DIRS=()

log_info "총 ${TOTAL_DIRS}개의 디렉토리를 처리합니다."

# 각 디렉토리별로 처리
for video_dir in "${VIDEO_DIRS[@]}"; do
    CURRENT_DIR=$((CURRENT_DIR + 1))
    
    # 디렉토리 존재 여부 확인
    if [ ! -d "$video_dir" ]; then
        log_warning "디렉토리가 존재하지 않습니다: $video_dir"
        FAILED_DIRS+=("$video_dir")
        continue
    fi
    
    log_info "[${CURRENT_DIR}/${TOTAL_DIRS}] 처리 중: $video_dir"
    
    # Python 스크립트 실행
    if python mediapipe_s3_streaming_pipeline.py \
        --video-dir "$video_dir" \
        --s3-bucket waterandfish-s3 \
        --s3-prefix sign-language-data \
        --target-fps 15 \
        --model-complexity 1 \
        --max-workers 2; then
        
        log_success "[${CURRENT_DIR}/${TOTAL_DIRS}] 완료: $video_dir"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log_error "[${CURRENT_DIR}/${TOTAL_DIRS}] 실패: $video_dir"
        FAILED_DIRS+=("$video_dir")
    fi
    
    # 잠시 대기 (시스템 부하 방지)
    sleep 2
done

# 최종 결과 출력
echo ""
log_info "=== 처리 완료 ==="
log_success "성공: ${SUCCESS_COUNT}/${TOTAL_DIRS} 디렉토리"

if [ ${#FAILED_DIRS[@]} -gt 0 ]; then
    log_error "실패한 디렉토리 (${#FAILED_DIRS[@]}개):"
    for failed_dir in "${FAILED_DIRS[@]}"; do
        echo "  - $failed_dir"
    done
    exit 1
else
    log_success "모든 디렉토리가 성공적으로 처리되었습니다!"
fi 