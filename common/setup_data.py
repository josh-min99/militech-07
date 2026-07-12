"""
common/setup_data.py (1차 초안)

역할: 구글드라이브 zip을 풀어서 나온 원본 영상들을 모든 모델이 공유하는
      /workspace/data/real_military_dataset 구조로 정리하는 '파일 정리 스크립트'.
      특정 모델의 데이터로더 코드가 아니라, 데이터 준비 단계에서 1회만 실행됨.
      (여러 모델을 붙일 예정이라 SRC/DST를 특정 모델 폴더가 아닌 공용 data/ 경로로 고정)

      정상(Normal) 영상  -> data/real_military_dataset/Training/Normal/
      침입 포함 전체 영상 -> data/real_military_dataset/Validation/All/

주의: 실제 AI Hub zip을 풀었을 때 나오는 폴더/파일명 구조를 아직 확인 못 했음.
      아래 NORMAL_KEYWORDS 조건은 임시 규칙 — /workspace/data/_extracted 구조를
      `find /workspace/data/_extracted -maxdepth 3` 로 확인한 뒤 알려주면
      정확한 분류 조건으로 다시 다듬어줄 것.
"""
import os
import shutil
import glob

SRC = "/workspace/data/_extracted"
DST = "/workspace/data/real_military_dataset"

# TODO: 실제 zip 구조 확인 후 정확한 키워드/폴더 규칙으로 교체
NORMAL_KEYWORDS = ["정상", "Normal", "normal"]

train_dir = os.path.join(DST, "Training", "Normal")
val_dir = os.path.join(DST, "Validation", "All")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

videos = glob.glob(os.path.join(SRC, "**", "*.mp4"), recursive=True)
print(f"[setup_data] 총 {len(videos)}개 영상 발견")

n_train, n_val = 0, 0
for v in videos:
    is_normal = any(kw in v for kw in NORMAL_KEYWORDS)
    if is_normal:
        shutil.copy2(v, os.path.join(train_dir, os.path.basename(v)))
        n_train += 1
    else:
        shutil.copy2(v, os.path.join(val_dir, os.path.basename(v)))
        n_val += 1

print(f"[setup_data] Training/Normal 로 분류: {n_train}개")
print(f"[setup_data] Validation/All   로 분류: {n_val}개")
