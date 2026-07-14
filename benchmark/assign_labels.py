import os
import pandas as pd
import re
import numpy as np

# 데이터 루트 (환경변수 AI_DATA_ROOT 로 덮어쓸 수 있음, 기본 D:\AI_data)
DATA_ROOT = os.environ.get("AI_DATA_ROOT", r"D:\AI_data")

def assign_label(path: str) -> str:
    fname = os.path.splitext(os.path.basename(path))[0]   # 폴더 무관하게 파일명만
    action = fname.rsplit('_', 1)[-1]
    codes = re.findall(r'[A-E]\d', action)
    n = len(codes)
    if n == 1:
        return 'Normal'
    elif n == 2:
        return 'Approach'
    else:
        return 'Intrusion'

for csv_path in [os.path.join(DATA_ROOT, 'military_train.csv'),
                 os.path.join(DATA_ROOT, 'military_test.csv')]:
    df = pd.read_csv(csv_path)
    df['label'] = df['path'].apply(assign_label)
    df.to_csv(csv_path, index=False)
    print(csv_path)
    print(df['label'].value_counts().to_string())
    print()

# gt.npy 생성 (test용)
df_test = pd.read_csv(os.path.join(DATA_ROOT, 'military_test.csv'))
gt_labels = np.array([0 if l == 'Normal' else 1 for l in df_test['label']])

# 각 .npy 파일의 실제 프레임 수를 읽어서 gt를 정확하게 생성
# military_test.py는 각 시퀀스에서 len_cur개 예측 후 np.repeat(ap, 16) 수행
gt_frames = []
for i, row in df_test.iterrows():
    feat = np.load(row['path'])
    n_frames = feat.shape[0]        # 실제 프레임 수 (1~4)
    gt_frames.extend([gt_labels[df_test.index.get_loc(i)]] * (n_frames * 16))

gt_frame = np.array(gt_frames)
np.save(os.path.join(DATA_ROOT, 'gt_military.npy'), gt_frame)
np.save(os.path.join(DATA_ROOT, 'gt_segment_military.npy'),
        np.array([[i, i+1] for i in range(len(df_test))], dtype=object))
np.save(os.path.join(DATA_ROOT, 'gt_label_military.npy'),
        np.array(df_test['label'].tolist(), dtype=object))
print(f"gt files saved ({len(gt_frame)} frames total)")
