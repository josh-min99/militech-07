import pandas as pd
import re
import numpy as np

def assign_label(path: str) -> str:
    fname = path.replace('D:\\AI_data\\features\\', '').replace('.npy', '')
    action = fname.rsplit('_', 1)[-1]
    codes = re.findall(r'[A-E]\d', action)
    n = len(codes)
    if n == 1:
        return 'Normal'
    elif n == 2:
        return 'Approach'
    else:
        return 'Intrusion'

for csv_path in [r'D:\AI_data\military_train.csv', r'D:\AI_data\military_test.csv']:
    df = pd.read_csv(csv_path)
    df['label'] = df['path'].apply(assign_label)
    df.to_csv(csv_path, index=False)
    print(csv_path)
    print(df['label'].value_counts().to_string())
    print()

# gt.npy 생성 (test용)
df_test = pd.read_csv(r'D:\AI_data\military_test.csv')
gt_labels = np.array([0 if l == 'Normal' else 1 for l in df_test['label']])

# 각 .npy 파일의 실제 프레임 수를 읽어서 gt를 정확하게 생성
# military_test.py는 각 시퀀스에서 len_cur개 예측 후 np.repeat(ap, 16) 수행
gt_frames = []
for i, row in df_test.iterrows():
    feat = np.load(row['path'])
    n_frames = feat.shape[0]        # 실제 프레임 수 (1~4)
    gt_frames.extend([gt_labels[df_test.index.get_loc(i)]] * (n_frames * 16))

gt_frame = np.array(gt_frames)
np.save(r'D:\AI_data\gt_military.npy', gt_frame)
np.save(r'D:\AI_data\gt_segment_military.npy',
        np.array([[i, i+1] for i in range(len(df_test))], dtype=object))
np.save(r'D:\AI_data\gt_label_military.npy',
        np.array(df_test['label'].tolist(), dtype=object))
print(f"gt files saved ({len(gt_frame)} frames total)")
