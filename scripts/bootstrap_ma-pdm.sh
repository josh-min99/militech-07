#!/bin/bash
# scripts/bootstrap_ma-pdm.sh (militech-07 저장소 내부 경로)
#
# 사용법 (Vast.ai 인스턴스에서):
#   git clone https://github.com/josh-min99/militech-07.git
#   bash militech-07/scripts/bootstrap_ma-pdm.sh
#
# 데이터(data/real_military_dataset)는 모든 모델이 공유하므로,
# 이미 존재하면 다운로드를 건너뛰고 adapter 파일 복사 + 패키지 설치만 수행.
set -e

REPO_MODEL="https://github.com/henrryzh1/MA-PDM.git"
DRIVE_FOLDER="https://drive.google.com/drive/folders/18COtRvOVeU0kHKPSWDVpuD65XpA5IpL0"
WORKDIR="/workspace"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # militech-07 루트
ADAPTER_DIR="$REPO_ROOT/adapters/MA-PDM"
COMMON_DIR="$REPO_ROOT/common"
MODEL_DIR="$WORKDIR/models/MA-PDM"
DATA_DIR="$WORKDIR/data"

echo "=== [1/6] MA-PDM 베이스 클론 ==="
mkdir -p "$WORKDIR/models"
if [ ! -d "$MODEL_DIR" ]; then
    git clone "$REPO_MODEL" "$MODEL_DIR"
fi

echo "=== [2/5] adapter 파일(설정+데이터로더) 덮어쓰기 ==="
# MA-PDM 원본이 기본적으로 'configs'(복수) 폴더를 읽으므로 이름을 그대로 맞춤 -> 코드 패치 불필요
mkdir -p "$MODEL_DIR/configs"
cp "$ADAPTER_DIR/configs/military.yml" "$MODEL_DIR/configs/military.yml"
cp "$ADAPTER_DIR/datasets/military.py" "$MODEL_DIR/datasets/military.py"
cp "$ADAPTER_DIR/datasets/__init__.py" "$MODEL_DIR/datasets/__init__.py"
cp "$COMMON_DIR/setup_data.py"         "$MODEL_DIR/setup_data.py"

cd "$MODEL_DIR"

echo "=== [3/5] 패키지 설치 ==="
pip install -q opencv-python-headless ipdb tqdm tensorboardX pyyaml gdown

echo "=== [4/5] 데이터 준비 (공용 data/ 폴더, 이미 있으면 스킵) ==="
if [ -d "$DATA_DIR/real_military_dataset/Training/Normal" ] && [ -n "$(ls -A "$DATA_DIR/real_military_dataset/Training/Normal" 2>/dev/null)" ]; then
    echo "이미 정리된 공용 데이터 발견 -> 다운로드/정리 스킵 (다른 모델이 준비해둔 데이터 재사용)"
else
    mkdir -p "$DATA_DIR/_gdrive_raw" "$DATA_DIR/_extracted"
    cd "$DATA_DIR/_gdrive_raw"
    gdown --folder "$DRIVE_FOLDER"
    find "$DATA_DIR/_gdrive_raw" -name "*.zip" -exec unzip -oq {} -d "$DATA_DIR/_extracted" \;
    cd "$MODEL_DIR"
    python3 setup_data.py
fi

echo ""
echo "=== [5/5] 준비 완료 ==="
echo "Training/Normal 영상 수: $(ls "$DATA_DIR/real_military_dataset/Training/Normal" 2>/dev/null | wc -l)"
echo "다음 명령으로 바로 학습 시작:"
echo "  tmux new -s train_mapdm"
echo "  cd $MODEL_DIR && python3 train_diffusion.py --config military.yml"
