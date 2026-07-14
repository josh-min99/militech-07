"""
군 경계 데이터셋용 CLIP 특징 추출 파이프라인.

[사용 방법]
  python extract_military_features.py

[지원 모드]
  Mode 1 (frame_data): D:/AI_data/Frame_data 의 JPG 이미지 → .npy 특징 파일 생성
  Mode 2 (label_data): D:/AI_data/Labeling_data JSON 라벨 + 이미지(별도 경로) → 라벨 포함 .npy 생성

결과물:
  D:/AI_data/features/{sensor}_{action}.npy   (각 시퀀스의 CLIP 특징)
  D:/AI_data/military_train.csv
  D:/AI_data/military_test.csv
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

import _env  # noqa: F401  — VadCLIP src를 sys.path에 추가 (clip 패키지 포함)
from clip import clip

# ── 경로 설정 (환경변수 AI_DATA_ROOT 로 덮어쓸 수 있음, 기본 D:\AI_data) ─────────
DATA_ROOT   = os.environ.get("AI_DATA_ROOT", r"D:\AI_data")
FRAME_DATA  = os.path.join(DATA_ROOT, "Frame_data")
LABEL_DATA  = os.path.join(DATA_ROOT, "Labeling_data")
FEATURE_OUT = os.path.join(DATA_ROOT, "features")

# JSON class 필드 → VadCLIP 라벨 매핑
# class 0: 정상 해양 활동 / 1: 일반 선박 접근 / 2: 비인가 활동 / 4: 위협
CLASS_TO_LABEL = {
    '0': 'Normal',
    '1': 'Approach',
    '2': 'Intrusion',
    '4': 'Threat',
}

# ── 이미지 파일 → CLIP 특징 추출 ─────────────────────────────────────────────
def extract_clip_features(image_paths: list, model, preprocess, device) -> np.ndarray:
    features = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(tensor)            # [1, 512]
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)              # [N, 512]


# ── Mode 1: Frame_data (라벨 없음) ────────────────────────────────────────────
def parse_frame_filename(fname: str):
    """EO_SU_DT_W1_H1_A1A5A4_0001.jpg → (sensor, action, frame_num)"""
    name = os.path.splitext(fname)[0]
    parts = name.split('_')
    frame_num = parts[-1]
    action    = parts[-2]   # 행동 코드 (A1A5A4 등)
    prefix    = '_'.join(parts[:-2])   # EO_SU_DT_W1_H1
    return prefix, action, frame_num


def collect_frame_sequences(frame_data: str = None) -> dict:
    """Frame_data 폴더에서 (시퀀스 키 → 이미지 경로 목록) 수집.

    frame_data: 원본 이미지 루트. None이면 모듈 기본값(FRAME_DATA) 사용.
    """
    frame_data = frame_data or FRAME_DATA
    sequences = defaultdict(list)
    for root, _, files in os.walk(frame_data):
        for fname in files:
            if not fname.lower().endswith('.jpg'):
                continue
            prefix, action, _ = parse_frame_filename(fname)
            # 폴더 경로에서 상위 구분자 추출 (EO/SU/DT 등)
            rel = os.path.relpath(root, frame_data)          # EO\SU\DT
            seq_key = f"{rel.replace(os.sep, '_')}_{action}"
            sequences[seq_key].append(os.path.join(root, fname))
    for key in sequences:
        sequences[key].sort()
    return dict(sequences)


# ── Mode 2: Labeling_data (라벨 있음, 이미지는 별도 경로 필요) ────────────────
def collect_labeled_sequences(image_root: str) -> dict:
    """
    Labeling_data JSON을 읽어 이미지 경로 + 라벨을 수집.
    image_root: JSON 내 filename에 해당하는 이미지들이 있는 폴더
    """
    sequences = {}
    for dirpath, dirnames, files in os.walk(LABEL_DATA):
        json_files = sorted(f for f in files if f.endswith('.json'))
        if not json_files:
            continue
        seq_name = os.path.basename(dirpath)
        frame_paths = []
        label_counter = defaultdict(int)

        for jf in json_files:
            with open(os.path.join(dirpath, jf), encoding='utf-8') as fp:
                data = json.load(fp)
            ann = data['annotations'][0]
            img_name = ann['filename']
            img_path = os.path.join(image_root, img_name)
            if os.path.exists(img_path):
                frame_paths.append(img_path)
            cls = str(ann.get('class', '0'))
            label_counter[cls] += 1

        if not frame_paths:
            continue

        # 가장 많이 등장한 class가 해당 시퀀스의 라벨
        majority_class = max(label_counter, key=label_counter.get)
        label = CLASS_TO_LABEL.get(majority_class, 'Normal')
        sequences[seq_name] = {'paths': frame_paths, 'label': label}

    return sequences


# ── CSV 생성 ──────────────────────────────────────────────────────────────────
def write_csv(rows: list, out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("path,label\n")
        for row in rows:
            f.write(row + '\n')
    print(f"  CSV saved: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CLIP ViT-B/16...")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()

    os.makedirs(FEATURE_OUT, exist_ok=True)

    # ── Mode 1: Frame_data ──
    print("\n[Mode 1] extracting features from Frame_data images...")
    sequences = collect_frame_sequences()
    print(f"  sequences found: {len(sequences)}")

    train_rows, test_rows = [], []
    seq_list = sorted(sequences.items())
    split_idx = int(len(seq_list) * 0.8)   # 80% train / 20% test

    for i, (seq_key, image_paths) in enumerate(seq_list):
        print(f"  [{i+1}/{len(seq_list)}] {seq_key} ({len(image_paths)} imgs)")
        features = extract_clip_features(image_paths, model, preprocess, device)
        save_path = os.path.join(FEATURE_OUT, f"{seq_key}.npy")
        np.save(save_path, features)

        # Frame_data에는 라벨이 없으므로 모두 Normal로 초기화
        # 실제 라벨은 아래 label_map.txt를 편집해 재생성하세요
        label = 'Normal'
        row = f"{save_path},{label}"
        if i < split_idx:
            train_rows.append(row)
        else:
            test_rows.append(row)

    write_csv(train_rows, os.path.join(DATA_ROOT, "military_train.csv"))
    write_csv(test_rows,  os.path.join(DATA_ROOT, "military_test.csv"))

    # ── Mode 2: Labeling_data (이미지 파일이 있을 때) ──
    # Labeling_data 내 이미지가 없어서 현재는 건너뜀
    # 이미지 경로를 알게 되면 아래 줄의 주석을 해제하세요:
    #
    # IMAGE_ROOT = r"D:\AI_data\원천데이터"  # 실제 이미지 폴더로 변경
    # labeled = collect_labeled_sequences(IMAGE_ROOT)
    # rows = []
    # for seq_name, info in labeled.items():
    #     features = extract_clip_features(info['paths'], model, preprocess, device)
    #     save_path = os.path.join(FEATURE_OUT, f"labeled_{seq_name}.npy")
    #     np.save(save_path, features)
    #     rows.append(f"{save_path},{info['label']}")
    # write_csv(rows, os.path.join(DATA_ROOT, "military_labeled_train.csv"))

    print(f"\nDone. Features: {FEATURE_OUT}")
    print(f"  train: {len(train_rows)} sequences")
    print(f"  test:  {len(test_rows)} sequences")
    print("\n[Next steps]")
    print("  1. Edit the 'label' column in D:/AI_data/military_train.csv with real labels")
    print("  2. python military_train.py")


if __name__ == '__main__':
    main()
