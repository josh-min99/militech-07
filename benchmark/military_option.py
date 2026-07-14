import argparse
import os

# ── 경로 설정 (환경변수 > 기본값, 실행 시 CLI 플래그로도 덮어쓸 수 있음) ──────────
#   AI_DATA_ROOT        : 데이터/결과 루트 폴더  (기본 D:\AI_data)
#   MILITECH_MODEL_PATH : 학습 가중치 경로       (기본 models/VadCLIP/src/model/model_military.pth)
_BENCH_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BENCH_DIR)            # militech-07
_MODEL_DIR    = os.path.join(_PROJECT_ROOT, "models", "VadCLIP", "src", "model")

# 다른 스크립트(benchmark.py, compare_models.py 등)가 import 해서 쓰는 공통 경로.
DATA_ROOT   = os.environ.get("AI_DATA_ROOT", r"D:\AI_data")
FRAME_DATA  = os.path.join(DATA_ROOT, "Frame_data")       # end2end 원본 이미지
FEATURE_DIR = os.path.join(DATA_ROOT, "features")         # 추출된 CLIP 특징
RESULT_DIR  = os.path.join(DATA_ROOT, "benchmark_results")  # 벤치마크 결과 JSON
_MODEL_PATH = os.environ.get("MILITECH_MODEL_PATH", os.path.join(_MODEL_DIR, "model_military.pth"))

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

# 파일 경로 (모두 CLI 플래그로 덮어쓸 수 있음)
parser.add_argument('--model-path',       default=_MODEL_PATH)
parser.add_argument('--checkpoint-path',  default=os.path.join(_MODEL_DIR, 'checkpoint_military.pth'))
parser.add_argument('--use-checkpoint',   default=False, type=bool)

parser.add_argument('--train-list', default=os.path.join(DATA_ROOT, 'military_train.csv'))
parser.add_argument('--test-list',  default=os.path.join(DATA_ROOT, 'military_test.csv'))

# 평가용 gt 파일 (extract_military_features.py 실행 후 생성됨)
parser.add_argument('--gt-path',         default=os.path.join(DATA_ROOT, 'gt_military.npy'))
parser.add_argument('--gt-segment-path', default=os.path.join(DATA_ROOT, 'gt_segment_military.npy'))
parser.add_argument('--gt-label-path',   default=os.path.join(DATA_ROOT, 'gt_label_military.npy'))
