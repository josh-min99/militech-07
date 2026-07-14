import argparse
import os

# 이 파일이 어디로 옮겨지든 VadCLIP 모델/가중치 위치를 절대경로로 계산한다.
_BENCH_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BENCH_DIR)            # militech-07
_MODEL_DIR    = os.path.join(_PROJECT_ROOT, "models", "VadCLIP", "src", "model")

parser = argparse.ArgumentParser(description='VadCLIP - Military Boundary Dataset')
parser.add_argument('--seed', default=234, type=int)

# 모델 구조 (CLIP ViT-B/16 기준 그대로 유지)
parser.add_argument('--embed-dim',     default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width',  default=512, type=int)
parser.add_argument('--visual-head',   default=1,   type=int)
parser.add_argument('--visual-layers', default=2,   type=int)
parser.add_argument('--attn-window',   default=8,   type=int)
parser.add_argument('--prompt-prefix', default=10,  type=int)
parser.add_argument('--prompt-postfix',default=10,  type=int)

# 군 경계 데이터셋 클래스 수: Normal + Approach + Intrusion + Threat = 4
parser.add_argument('--classes-num',   default=4,   type=int)

# 학습 설정
parser.add_argument('--max-epoch',     default=10,  type=int)
parser.add_argument('--batch-size',    default=4,   type=int)
parser.add_argument('--lr',            default=2e-5)
parser.add_argument('--scheduler-rate',       default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

# 파일 경로
parser.add_argument('--model-path',       default=os.path.join(_MODEL_DIR, 'model_military.pth'))
parser.add_argument('--checkpoint-path',  default=os.path.join(_MODEL_DIR, 'checkpoint_military.pth'))
parser.add_argument('--use-checkpoint',   default=False, type=bool)

parser.add_argument('--train-list', default=r'D:\AI_data\military_train.csv')
parser.add_argument('--test-list',  default=r'D:\AI_data\military_test.csv')

# 평가용 gt 파일 (extract_military_features.py 실행 후 생성됨)
parser.add_argument('--gt-path',         default=r'D:\AI_data\gt_military.npy')
parser.add_argument('--gt-segment-path', default=r'D:\AI_data\gt_segment_military.npy')
parser.add_argument('--gt-label-path',   default=r'D:\AI_data\gt_label_military.npy')
