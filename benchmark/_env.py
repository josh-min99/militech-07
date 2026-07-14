"""
벤치마크 스크립트가 models/VadCLIP/src 의 모델 코드(model.py, utils/, clip/)를
import 할 수 있도록 sys.path 에 해당 경로를 추가한다.

`import _env` 한 줄이면 경로 설정이 끝난다.
기존 VadCLIP 코드는 전혀 수정하지 않는다 — 오직 참조(import)만 한다.

다른 모델을 벤치마크할 때는 아래 VADCLIP_SRC 대신 그 모델의 소스 경로를
sys.path 에 추가하도록 이 파일(또는 새 _env)만 바꾸면 된다.
"""
import os
import sys

# 콘솔 출력 인코딩을 UTF-8로 고정 (Windows 콘솔에서 깨짐 방지용 보조 장치).
# 결과/출력 문자열 자체는 ASCII(영어)로 통일하지만, 이중 안전장치로 둔다.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

_BENCH_DIR    = os.path.dirname(os.path.abspath(__file__))   # militech-07/benchmark
_PROJECT_ROOT = os.path.dirname(_BENCH_DIR)                   # militech-07
VADCLIP_SRC   = os.path.join(_PROJECT_ROOT, "models", "VadCLIP", "src")

if VADCLIP_SRC not in sys.path:
    sys.path.insert(0, VADCLIP_SRC)
