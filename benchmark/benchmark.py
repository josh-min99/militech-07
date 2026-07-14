"""
모델 성능 + 추론 속도 벤치마크 (군 경계 데이터셋).

측정 범위(scope)를 두 가지로 명확히 구분한다.
  - inference : 이미 추출된 특징(feature) -> 이상 점수. 모델 자체의 순수 추론 속도.
  - end2end   : 원본 이미지 -> 특징 추출(CLIP) -> 이상 점수. 실제 배포 파이프라인 전체.

지연시간 측정 표준은 아래 [TIMING HARNESS] 블록에 고정되어 있으며,
모델이 바뀌어도 이 블록은 절대 수정하지 않는다. (BENCHMARK.md 참고)

사용법:
  python benchmark.py --mode inference
  python benchmark.py --mode end2end
  python benchmark.py --mode both --model-name VadCLIP
"""

import time
import json
import os
import argparse
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime

import _env  # noqa: F401  — VadCLIP src(model.py, utils/, clip/)를 sys.path에 추가
from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text, process_split
import military_option
from extract_military_features import collect_frame_sequences

LABEL_MAP = {
    'Normal':    'normal maritime surveillance activity',
    'Approach':  'vessel approaching restricted military zone',
    'Intrusion': 'unauthorized vessel intrusion detected',
    'Threat':    'critical maritime security threat',
}

RESULT_DIR   = r"D:\AI_data\benchmark_results"
WARMUP_ITERS = 10          # 워밍업 반복 횟수 (측정에서 제외) — 모든 모델 공통
TIMER_DESC   = "time.perf_counter() + torch.cuda.synchronize()"


# ══════════════════════════════════════════════════════════════════════════════
#  [TIMING HARNESS]  ── 절대 수정 금지 ──
#  모델/스테이지가 무엇이든 이 두 함수로만 시간을 측정한다.
#  이것이 "모든 모델에서 측정 방법과 시점이 동일함"을 보장하는 핵심이다.
# ══════════════════════════════════════════════════════════════════════════════
def _sync(device):
    """GPU 비동기 실행을 강제로 완료시켜 정확한 wall-clock 측정을 보장."""
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, device):
    """
    fn()을 1회 실행하고 소요 시간(ms)과 반환값을 돌려준다.
    측정 경계:
      - 시작: _sync 직후 t0
      - 끝  : fn 반환 + _sync 직후 t1
    즉 GPU 커널이 실제로 끝난 시점까지 포함한다.
    """
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out
# ══════════════════════════════════════════════════════════════════════════════
#  [/TIMING HARNESS]
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  [MODEL ADAPTER]  ── 새 모델을 추가할 때는 이 클래스를 상속해 구현한다 ──
#  구현해야 하는 4가지:
#    load()            : 추론 모델 로드
#    load_extractor()  : (end2end 전용) 특징 추출기 로드
#    extract_closure() : 이미지 경로 리스트 -> 특징 ndarray [N, D] 를 만드는 무인자 함수 반환
#    infer_closure()   : 준비된 입력 -> per-frame 이상 점수(dict) 를 만드는 무인자 함수 반환
#    prepare_infer()   : 특징 ndarray -> 모델 입력 텐서(디바이스 위)  (측정에서 제외되는 glue)
# ══════════════════════════════════════════════════════════════════════════════
class ModelAdapter:
    name = "BaseAdapter"
    feature_dim = None

    def load(self):                       raise NotImplementedError
    def load_extractor(self):             raise NotImplementedError
    def extract_closure(self, image_paths): raise NotImplementedError
    def prepare_infer(self, feat_np):     raise NotImplementedError
    def infer_closure(self, prepared):    raise NotImplementedError
    def num_parameters(self) -> int:      raise NotImplementedError


class VadCLIPAdapter(ModelAdapter):
    name = "VadCLIP"
    feature_dim = 512

    def __init__(self, args, device):
        self.args        = args
        self.device      = device
        self.maxlen      = args.visual_length
        self.prompt_text = get_prompt_text(LABEL_MAP)
        self.model       = None
        self.clip_model  = None
        self.preprocess  = None

    # ── 추론 모델 로드 ──────────────────────────────────────────────
    def load(self):
        m = CLIPVAD(
            self.args.classes_num, self.args.embed_dim, self.args.visual_length,
            self.args.visual_width, self.args.visual_head, self.args.visual_layers,
            self.args.attn_window, self.args.prompt_prefix, self.args.prompt_postfix,
            self.device,
        )
        m.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        m.to(self.device).eval()
        self.model = m

    # ── 특징 추출기(CLIP) 로드 (end2end 전용) ──────────────────────
    def load_extractor(self):
        from clip import clip
        self.clip_model, self.preprocess = clip.load("ViT-B/16", self.device)
        self.clip_model.eval()

    # ── [Scope B] 이미지 -> 특징 ndarray [N, 512] ─────────────────
    def extract_closure(self, image_paths):
        def run():
            feats = []
            with torch.no_grad():
                for p in image_paths:
                    img = Image.open(p).convert("RGB")
                    t   = self.preprocess(img).unsqueeze(0).to(self.device)
                    feats.append(self.clip_model.encode_image(t))
            return torch.cat(feats, dim=0).float().cpu().numpy()
        return run

    # ── glue: 특징 ndarray -> 모델 입력 (측정 제외) ────────────────
    def prepare_infer(self, feat_np):
        feat, length = process_split(feat_np, self.maxlen)
        length  = int(length)
        len_cur = length
        visual  = torch.tensor(feat)
        if len_cur < self.maxlen:
            visual = visual.unsqueeze(0)
        visual  = visual.to(self.device)

        n = int(length / self.maxlen) + 1
        lengths = torch.zeros(n)
        rem = length
        for j in range(n):
            lengths[j] = min(rem, self.maxlen)
            rem -= self.maxlen
            if rem <= 0:
                break
        lengths = lengths.to(int)
        padding_mask = get_batch_mask(lengths, self.maxlen).to(self.device)
        return {"visual": visual, "padding_mask": padding_mask,
                "lengths": lengths, "len_cur": len_cur}

    # ── [Scope A] 모델 입력 -> per-frame 이상 점수 ────────────────
    def infer_closure(self, prepared):
        def run():
            with torch.no_grad():
                _, logits1, logits2 = self.model(
                    prepared["visual"], prepared["padding_mask"],
                    self.prompt_text, prepared["lengths"])
                logits1 = logits1.reshape(-1, logits1.shape[2])
                logits2 = logits2.reshape(-1, logits2.shape[2])
                lc = prepared["len_cur"]
                prob1 = torch.sigmoid(logits1[0:lc].squeeze(-1))
                prob2 = 1 - logits2[0:lc].softmax(dim=-1)[:, 0]
                return {"prob1": prob1.detach().cpu().numpy(),
                        "prob2": prob2.detach().cpu().numpy()}
        return run

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
# ══════════════════════════════════════════════════════════════════════════════
#  [/MODEL ADAPTER]
# ══════════════════════════════════════════════════════════════════════════════


# ── 통계 헬퍼 ──────────────────────────────────────────────────────────────────
def latency_stats(ms_list: list) -> dict:
    a = np.asarray(ms_list, dtype=np.float64)
    return {
        "mean_ms":   round(float(a.mean()), 3),
        "median_ms": round(float(np.median(a)), 3),
        "p95_ms":    round(float(np.percentile(a, 95)), 3),
        "min_ms":    round(float(a.min()), 3),
        "max_ms":    round(float(a.max()), 3),
    }


# ── 한 샘플 측정: 스테이지별 시간 + 점수 ───────────────────────────────────────
def measure_sample(adapter, sample, mode, device):
    times = {}
    if mode == "end2end":
        ex_ms, feat_np = time_call(adapter.extract_closure(sample["image_paths"]), device)
        times["extract_ms"] = ex_ms
    else:
        feat_np = sample["feat_np"]

    prepared      = adapter.prepare_infer(feat_np)          # glue (측정 제외)
    inf_ms, out   = time_call(adapter.infer_closure(prepared), device)
    times["infer_ms"] = inf_ms
    return times, out


# ── 한 scope 전체 실행 (warmup 포함) ───────────────────────────────────────────
def run_pass(adapter, samples, mode, device):
    # 워밍업 (첫 샘플로 WARMUP_ITERS회, 측정에서 제외)
    for _ in range(WARMUP_ITERS):
        measure_sample(adapter, samples[0], mode, device)

    extract_ms, infer_ms = [], []
    prob1_all, prob2_all, gt_all = [], [], []

    for s in samples:
        times, out = measure_sample(adapter, s, mode, device)
        infer_ms.append(times["infer_ms"])
        if "extract_ms" in times:
            extract_ms.append(times["extract_ms"])

        prob1_all.append(out["prob1"])
        prob2_all.append(out["prob2"])
        y = 0 if s["label"] == "Normal" else 1
        gt_all.append(np.full(len(out["prob1"]), y))

    prob1 = np.concatenate(prob1_all)
    prob2 = np.concatenate(prob2_all)
    gt    = np.concatenate(gt_all)

    metrics = {
        "AUC1": round(roc_auc_score(gt, prob1), 4),
        "AP1":  round(average_precision_score(gt, prob1), 4),
        "AUC2": round(roc_auc_score(gt, prob2), 4),
        "AP2":  round(average_precision_score(gt, prob2), 4),
    }
    return metrics, infer_ms, extract_ms


# ── 샘플 목록 로드 ─────────────────────────────────────────────────────────────
def load_samples(test_list, need_images):
    import pandas as pd
    df = pd.read_csv(test_list)
    seq_map = collect_frame_sequences() if need_images else {}
    samples, skipped = [], 0
    for _, row in df.iterrows():
        path    = row["path"]
        label   = row["label"]
        seq_key = os.path.splitext(os.path.basename(path))[0]
        s = {"label": label, "feat_np": np.load(path), "seq_key": seq_key}
        if need_images:
            imgs = seq_map.get(seq_key)
            if not imgs:
                skipped += 1
                continue
            s["image_paths"] = imgs
        samples.append(s)
    if skipped:
        print(f"  [WARN] {skipped} sequences skipped (source images not matched) for end2end")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='inference',
                    choices=['inference', 'end2end', 'both'],
                    help='inference=features->scores / end2end=images->scores / both=both')
    ap.add_argument('--model-name', default='VadCLIP')
    ap.add_argument('--note', default='')
    args_cli, _ = ap.parse_known_args()

    base = military_option.parser.parse_args([])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = VadCLIPAdapter(base, device)
    print(f"[{adapter.name}] loading inference model...")
    adapter.load()

    need_e2e = args_cli.mode in ("end2end", "both")
    if need_e2e:
        print(f"[{adapter.name}] loading feature extractor (CLIP)...")
        adapter.load_extractor()

    record = {
        "model_name": args_cli.model_name,
        "timestamp":  datetime.now().isoformat(timespec='seconds'),
        "device":     device,
        "gpu":        torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dataset":    "military_boundary",
        "label_note": args_cli.note or "action-code heuristic (Normal/Approach/Intrusion)",
        "protocol": {
            "timer":        TIMER_DESC,
            "warmup_iters": WARMUP_ITERS,
            "batch_size":   1,
            "scope_A_inference": "feature tensor (ready on device) -> per-frame scores. Pure model forward + scoring. Excludes disk I/O, feature extraction, input-tensor build.",
            "scope_B_end2end":   "raw image files -> extract(features) + infer. Input-tensor build (glue) is excluded.",
        },
        "model_info": {
            "parameters":        adapter.num_parameters(),
            "model_path":        base.model_path,
            "feature_extractor": "CLIP ViT-B/16",
            "feature_dim":       adapter.feature_dim,
            "classes_num":       base.classes_num,
        },
    }

    # ── inference scope ──
    if args_cli.mode in ("inference", "both"):
        samples = load_samples(base.test_list, need_images=False)
        print(f"[inference] measuring {len(samples)} sequences...")
        metrics, infer_ms, _ = run_pass(adapter, samples, "inference", device)
        total_s = sum(infer_ms) / 1000.0
        record["metrics"] = metrics
        record["speed_inference"] = {
            **latency_stats(infer_ms),
            "throughput_clips_per_s": round(len(infer_ms) / total_s, 1),
            "num_clips": len(infer_ms),
        }

    # ── end2end scope ──
    if args_cli.mode in ("end2end", "both"):
        samples = load_samples(base.test_list, need_images=True)
        print(f"[end2end] measuring {len(samples)} sequences...")
        metrics_e, infer_ms, extract_ms = run_pass(adapter, samples, "end2end", device)
        total_ms  = [e + i for e, i in zip(extract_ms, infer_ms)]
        num_imgs  = sum(len(s["image_paths"]) for s in samples)
        total_ex_s = sum(extract_ms) / 1000.0
        record.setdefault("metrics", metrics_e)
        record["speed_end2end"] = {
            "extract_mean_ms": round(float(np.mean(extract_ms)), 3),
            "infer_mean_ms":   round(float(np.mean(infer_ms)), 3),
            "total": latency_stats(total_ms),
            "image_throughput_fps": round(num_imgs / total_ex_s, 1),
            "num_clips":  len(total_ms),
            "num_images": num_imgs,
        }

    os.makedirs(RESULT_DIR, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RESULT_DIR, f"{args_cli.model_name}_{args_cli.mode}_{ts}.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # ── 요약 출력 ──
    print("\n" + "=" * 52)
    print(f"  Model: {record['model_name']}  ({record['gpu']})")
    m = record.get("metrics", {})
    print(f"  AUC1={m.get('AUC1')}  AP1={m.get('AP1')}  AUC2={m.get('AUC2')}  AP2={m.get('AP2')}")
    if "speed_inference" in record:
        si = record["speed_inference"]
        print(f"  [inference] {si['mean_ms']} ms/clip (p95 {si['p95_ms']}) | {si['throughput_clips_per_s']} clips/s")
    if "speed_end2end" in record:
        se = record["speed_end2end"]
        print(f"  [end2end]   extract {se['extract_mean_ms']} + infer {se['infer_mean_ms']} = {se['total']['mean_ms']} ms/clip")
        print(f"              extraction throughput {se['image_throughput_fps']} img/s")
    print("=" * 52)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
