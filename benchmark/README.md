# 모델 벤치마크 가이드 (군 경계 데이터셋)

이 문서는 **여러 이상탐지 모델을 서로 공정하게 비교**하기 위한 벤치마크 규격과 실행 방법을 정의한다.
성능 지표(AUC/AP)와 **추론 속도(지연시간)** 를 함께 측정하며, 모델이 달라져도 **측정 기준이 항상 동일하도록** 규칙을 고정한다.

### 폴더 구조

이 벤치마크 도구는 프로젝트 루트의 **`benchmark/`** 폴더에 모여 있으며,
**모델 코드(`models/VadCLIP/src/`)는 전혀 수정하지 않고 import(참조)만 한다.**

```
militech-07/
├─ benchmark/                     ← 벤치마크 도구 (이 폴더)
│  ├─ _env.py                     VadCLIP src 를 sys.path 에 추가 (경로 연결)
│  ├─ benchmark.py                벤치마크 실행 (측정 하네스 + 모델 어댑터)
│  ├─ compare_models.py           저장된 결과를 표로 비교
│  ├─ military_option.py          하이퍼파라미터·경로 설정
│  ├─ military_train.py           학습 스크립트
│  ├─ military_test.py            단독 테스트 스크립트
│  ├─ extract_military_features.py  이미지 -> CLIP 특징 추출
│  ├─ assign_labels.py            라벨 할당 + gt 파일 생성
│  └─ BENCHMARK.md                (이 문서)
└─ models/VadCLIP/src/            ← 원본 모델 코드 (수정 안 함)
   ├─ model.py, utils/, clip/     benchmark 가 import 하는 대상
   └─ model/model_military.pth    학습된 가중치 (여기 저장됨)
```

- 결과 저장 위치: `D:\AI_data\benchmark_results\*.json`
- 경로 연결은 `_env.py` 가 담당한다. 모든 스크립트가 `import _env` 한 줄로
  `models/VadCLIP/src` 를 `sys.path` 에 추가하므로, 벤치마크 파일이 어디에 있든 동작한다.

> **모델 코드 수정 필요 여부: 없음.**
> benchmark 는 `model.py`·`utils/`·`clip/` 를 그대로 import 만 하며, 원본 파일을 바꾸지 않는다.
> 학습 가중치는 `models/VadCLIP/src/model/` 에 두고, benchmark 는 절대경로로 이를 참조한다.

---

## 1. 빠른 시작

```bash
# 가상환경 파이썬 경로 (Windows / RTX)
#   ..\models\VadCLIP\.venv\Scripts\python.exe

cd benchmark

# (1) 추론 전용 속도 + 성능
python benchmark.py --mode inference --model-name VadCLIP

# (2) 원본 이미지부터 전 과정(end2end)
python benchmark.py --mode end2end  --model-name VadCLIP

# (3) 둘 다 한번에
python benchmark.py --mode both --model-name VadCLIP --note "memo"

# (4) 여러 모델 결과 비교
python compare_models.py                # AUC1 기준 정렬
python compare_models.py --sort infer   # 추론 지연 기준
python compare_models.py --sort e2e     # end2end 지연 기준
```

각 실행은 `D:\AI_data\benchmark_results\{모델명}_{모드}_{시각}.json` 파일 1개를 남긴다.
`compare_models.py` 는 이 폴더의 모든 JSON을 모아 표로 보여준다.

> 콘솔 출력과 결과 파일 텍스트는 **모두 영어(ASCII)** 로 통일했다 (Windows 콘솔 인코딩 깨짐 방지).
> `note` 도 영어로 쓰는 것을 권장한다.

---

## 2. 측정 범위(Scope) — 무엇을 재는가

추론 속도는 **어디부터 어디까지를 재느냐**에 따라 값이 완전히 달라진다.
따라서 두 가지 범위를 **명확히 분리**해서 측정한다.

| 모드 (`--mode`) | 측정 구간 | 용도 |
|---|---|---|
| `inference` | **이미 추출된 특징(feature) → 이상 점수** | 모델 아키텍처 자체의 순수 추론 속도 비교 |
| `end2end`   | **원본 이미지 → 특징 추출 → 이상 점수** | 실제 배포 파이프라인 전체 지연 |

> ⚠️ **중요:** `inference` 값만 보고 "이 모델은 10ms면 실시간 가능"이라고 결론내면 안 된다.
> 실제로는 특징 추출(예: CLIP 인코딩)이 대부분의 시간을 차지한다.
> VadCLIP 실측: 추론 17ms vs **특징추출 128ms** → end2end 145ms.

### 스테이지 경계 정의 (정확한 시점)

측정은 아래 두 스테이지로만 나뉘며, 각 스테이지의 **시작/끝 시점을 고정**한다.

| 스테이지 | 시작 시점 | 끝 시점 | 포함되는 것 | 제외되는 것 |
|---|---|---|---|---|
| **extract** (Scope B 전용) | 이미지 경로 리스트가 주어진 직후 | 특징 ndarray `[N, D]` 반환 + GPU 동기화 | 이미지 디코딩, 전처리(resize/normalize), 특징추출기 forward, GPU→CPU 복사 | 디스크 탐색, 파일 목록 수집 |
| **infer** (양쪽 공통) | 모델 입력 텐서가 **디바이스 위에 준비 완료된** 직후 | per-frame 이상 점수(host) 반환 + GPU 동기화 | 모델 forward, 점수화(softmax/sigmoid), GPU→CPU 복사 | 디스크 I/O, 특징추출, 입력 텐서 생성(`prepare_infer`) |

- **glue(측정 제외):** `prepare_infer` 에서 하는 텐서 변환·패딩·`to(device)` 는 두 모드 모두에서 **측정하지 않는다.**
  이 부분은 모델 구현마다 형태가 달라 공정 비교를 해치기 때문이며, 값도 무시할 수준이다.
- `end2end` 총 지연 = `extract` + `infer` (glue 제외).

---

## 3. 지연시간 측정 표준 — 모든 모델에서 동일하게

> 이 절의 규칙은 **모델이 바뀌어도 절대 변하지 않는다.** 이것이 공정 비교의 핵심이다.

`benchmark.py` 상단의 `[TIMING HARNESS]` 블록(`_sync`, `time_call`)이 이 규칙을 코드로 고정한다.
**새 모델을 추가할 때 이 블록은 복사만 하고 수정하지 않는다.**

| 항목 | 표준 | 이유 |
|---|---|---|
| 타이머 | `time.perf_counter()` | 고해상도 단조 증가 시계 |
| GPU 동기화 | 측정 시작 전과 끝에 `torch.cuda.synchronize()` | GPU 커널은 비동기 실행됨. 동기화 없으면 실제보다 짧게 측정됨 |
| 워밍업 | 첫 샘플로 **10회** 선실행 후 버림 (`WARMUP_ITERS=10`) | 초기 CUDA 컨텍스트/캐시/클럭 상승 구간 제외 |
| 배치 크기 | **1** | 지연(latency)은 단일 샘플 기준. 처리량과 구분 |
| 측정 단위 | **시퀀스(클립) 1개당** ms | 데이터셋 전체가 아닌 개별 샘플 기준 |
| 반복/집계 | 전체 테스트셋을 1회 순회, 샘플별 시간 수집 | 평균만이 아니라 분포를 봄 |
| 보고 통계 | `mean / median / p95 / min / max` | p95로 최악 지연(꼬리) 확인 |
| `torch.no_grad()` | 항상 적용 | 추론이므로 그래디언트 불필요 |
| `model.eval()` | 항상 적용 | dropout/BN 등 추론 모드 고정 |

### 측정 코드 (그대로 복사해서 사용)

```python
def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()

def time_call(fn, device):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()              # 측정 대상: extract 또는 infer 클로저
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out   # ms
```

### 다른 프레임워크에서 재사용할 때 — `_sync` 만 교체

`time_call` 의 구조(`sync → t0 → fn → sync → t1`)는 **절대 바꾸지 않는다.**
바꾸는 것은 오직 `_sync` 안의 **동기화 방법 한 줄**뿐이다. 프레임워크마다 GPU가
비동기로 도는 방식이 다르므로, 그 프레임워크의 "작업 완료 대기" 방법으로만 바꾼다.

| 프레임워크 | `_sync` 에 넣을 것 |
|---|---|
| PyTorch (CUDA) | `torch.cuda.synchronize()` (현재 코드 그대로) |
| PyTorch (CPU) | no-op (동기 실행이라 불필요) |
| TensorFlow | 출력 텐서에 `.numpy()` 를 호출해 강제 동기화 |
| JAX | `out.block_until_ready()` |
| ONNX Runtime / TensorRT | `session.run()` 이 이미 동기 → no-op |

> ⚠️ **타이머가 같다고 공정 비교가 보장되는 게 아니다.**
> 진짜 관건은 **`fn()`(클로저) 안에 무엇을 넣느냐** = Scope 경계(2절)다.
> 예: A모델은 `infer` 클로저에 `.cpu().numpy()` 를 넣고 B모델은 뺀다면, 타이머 코드가
> 똑같아도 B가 부당하게 빨라 보인다. **모든 모델이 2절의 스테이지 경계를 동일하게 지켜야**
> 비로소 숫자가 비교 가능하다. `time_call` 은 "동일하게 재는 자(尺)"일 뿐이고,
> "무엇을 잴지"는 어댑터 구현자가 규칙(2절)을 지켜 보장한다.

---

## 4. 처리량(Throughput)/FPS 정의 — 정직하게

- `throughput_clips_per_s` (inference) = 클립 수 ÷ 총 추론 시간 = `1000 / mean_ms`
- `image_throughput_fps` (end2end) = 원본 이미지 수 ÷ 총 특징추출 시간

> **주의:** "1개 특징 = N개 프레임"이라고 임의로 곱해 FPS를 부풀리지 않는다.
> 군 경계 데이터셋은 **1개 CLIP 특징 = 1장 이미지**이므로 곱셈 계수는 1이다.
> (원논문 UCF-Crime은 16프레임 스니펫당 1특징이라 ×16을 썼지만, 여기선 해당 없음.)

---

## 5. 결과 JSON 스키마

```jsonc
{
  "model_name": "VadCLIP",
  "timestamp":  "2026-07-14T20:15:08",
  "device":     "cuda",
  "gpu":        "NVIDIA GeForce RTX 4060",
  "dataset":    "military_boundary",
  "label_note": "...",
  "protocol": {                        // 측정 규격 (재현성 근거)
    "timer": "time.perf_counter() + torch.cuda.synchronize()",
    "warmup_iters": 10,
    "batch_size": 1,
    "scope_A_inference": "...",
    "scope_B_end2end":   "..."
  },
  "model_info": {
    "parameters": 162263043,
    "feature_extractor": "CLIP ViT-B/16",
    "feature_dim": 512
  },
  "metrics": { "AUC1":0.9519, "AP1":0.9946, "AUC2":0.9513, "AP2":0.9944 },
  "speed_inference": {                 // --mode inference / both
    "mean_ms":10.76, "median_ms":9.9, "p95_ms":15.05, "min_ms":..., "max_ms":...,
    "throughput_clips_per_s":92.9, "num_clips":73
  },
  "speed_end2end": {                   // --mode end2end / both
    "extract_mean_ms":127.78, "infer_mean_ms":16.91,
    "total": { "mean_ms":144.70, "median_ms":..., "p95_ms":..., ... },
    "image_throughput_fps":30.8, "num_clips":73, "num_images":287
  }
}
```

`protocol` 블록을 결과에 함께 저장하므로, 나중에 **어떤 기준으로 잰 값인지** 항상 확인할 수 있다.

---

## 6. 새 모델 추가 방법

새 모델은 `benchmark.py` 의 `ModelAdapter` 를 상속해 **어댑터만 구현**한다.
`[TIMING HARNESS]` 블록과 `run_pass`/`measure_sample` 등 **측정 로직은 건드리지 않는다.**
그리고 **원본 모델 코드 자체도 수정하지 않는다** — 아래처럼 경로만 연결한다.

### 경로 연결 (`_env.py`)

다른 모델의 소스가 `models/<YourModel>/src` 에 있다면, `_env.py` 에 그 경로를
`sys.path` 에 추가하는 줄을 더하면 된다 (또는 모델별 `_env` 를 새로 만든다).

```python
# _env.py 예시
YOUR_MODEL_SRC = os.path.join(_PROJECT_ROOT, "models", "YourModel", "src")
if YOUR_MODEL_SRC not in sys.path:
    sys.path.insert(0, YOUR_MODEL_SRC)
```

어댑터 파일 맨 위에서 `import _env` 한 줄이면 그 모델의 코드를 import 할 수 있다.
**이 방식이라 원본 모델 리포지토리는 한 줄도 고치지 않는다.**

### 구현해야 하는 메서드

```python
class MyModelAdapter(ModelAdapter):
    name = "MyModel"
    feature_dim = 512          # 특징 차원

    def load(self):
        """추론 모델을 self.model 에 로드하고 eval()."""

    def load_extractor(self):
        """(end2end 전용) 특징 추출기를 로드. inference만 할 거면 pass 가능."""

    def extract_closure(self, image_paths):
        """
        [Scope B] 무인자 함수를 반환.
        그 함수는 image_paths -> 특징 ndarray [N, feature_dim] 을 계산해 반환한다.
        (이미지 디코딩+전처리+추출기 forward 를 모두 포함할 것)
        """
        def run():
            ...
            return features_np
        return run

    def prepare_infer(self, feat_np):
        """
        glue(측정 제외): 특징 ndarray -> 모델 입력(디바이스 위 텐서/딕셔너리).
        여기서 len_cur(실제 프레임 수) 등 infer에 필요한 값도 함께 담는다.
        """
        return prepared

    def infer_closure(self, prepared):
        """
        [Scope A] 무인자 함수를 반환.
        그 함수는 prepared -> {"prob1": np[N], "prob2": np[N]} 를 반환한다.
        prob 는 per-frame 이상 점수(높을수록 이상). 단일 branch면 prob2=prob1 로 둔다.
        """
        def run():
            ...
            return {"prob1": scores1, "prob2": scores2}
        return run

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
```

그리고 `main()` 에서 어댑터를 교체:

```python
adapter = MyModelAdapter(base, device)   # <- VadCLIPAdapter 대신
```

### 출력 계약 (반드시 지킬 것)

- `infer_closure` 의 반환은 **per-frame 이상 점수**(길이 = 해당 클립의 프레임 수, 높을수록 이상).
- 라벨 기준: `Normal` → 정상(0), 그 외 → 이상(1). (AUC/AP 계산에 사용)
- 이 계약만 지키면 지표 계산·속도 측정·JSON 저장·비교표가 **자동으로 동일하게** 동작한다.

---

## 7. 공정 비교 체크리스트

여러 모델을 비교할 때 아래를 **동일하게** 유지해야 결과가 의미 있다.

- [ ] **같은 GPU/머신**에서 측정 (결과 JSON의 `gpu` 필드로 확인)
- [ ] **같은 테스트셋** (`military_test.csv` 동일 버전, 동일 시퀀스 목록)
- [ ] `--mode` 동일 (inference끼리, end2end끼리 비교)
- [ ] 워밍업 횟수·배치 크기 동일 (하네스가 강제 → 자동 보장)
- [ ] 백그라운드 부하 최소화 (다른 GPU 작업, 학습 동시 실행 금지)
- [ ] 가능하면 GPU 클럭 고정 (`nvidia-smi -lgc`) — 전력/온도에 따른 클럭 변동 억제
- [ ] 각 모델 **동일 정밀도**(fp32/fp16)로 측정하거나, 정밀도를 `note` 에 명시
- [ ] 특징 추출기가 다르면 end2end 비교 시 이를 `note`/`feature_extractor` 에 명시

---

## 8. 재현 환경 (현재 기준)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 |
| CUDA | 12.6 (드라이버) / torch cu124 |
| Python | 3.12 (`.venv`) |
| PyTorch | 2.6.0+cu124 |
| 특징 추출기 | CLIP ViT-B/16 (512-d) |
| 테스트셋 | military_test.csv (73 시퀀스, 287 이미지) |

---

## 9. 현재 VadCLIP 실측값 (참고)

| 지표 | 값 |
|---|---|
| AUC1 / AP1 | 0.9519 / 0.9946 |
| 추론 지연 (inference) | 10.8 ms/clip (p95 15.1) |
| end2end 지연 | 144.7 ms/clip (extract 127.8 + infer 16.9) |
| 특징추출 처리량 | 30.8 img/s |
| 파라미터 수 | 162,263,043 |

> 라벨이 파일명 휴리스틱 기반이라 성능 수치는 **파이프라인 검증용**이다.
> 실제 라벨(Labeling_data의 class 필드)이 확보되면 재학습 후 다시 벤치마크할 것.
