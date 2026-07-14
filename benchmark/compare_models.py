"""
D:/AI_data/benchmark_results/ 의 모든 결과 JSON을 읽어 비교 테이블 출력.

사용법:
  python compare_models.py                # 성능(AUC1) 기준 정렬
  python compare_models.py --sort infer   # 추론 지연 기준 정렬
  python compare_models.py --sort e2e     # end2end 지연 기준 정렬
"""

import json
import os
import argparse

import military_option

# 기본 결과 폴더 = military_option 의 값(환경변수 AI_DATA_ROOT 반영). --result-dir 로 덮어씀.
DEFAULT_RESULT_DIR = military_option.RESULT_DIR


def load_results(result_dir: str) -> list:
    results = []
    if not os.path.isdir(result_dir):
        return results
    for fname in sorted(os.listdir(result_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(result_dir, fname), encoding='utf-8') as f:
                results.append(json.load(f))
    return results


def get_infer_ms(r):
    return r.get("speed_inference", {}).get("mean_ms")


def get_e2e_ms(r):
    return r.get("speed_end2end", {}).get("total", {}).get("mean_ms")


def cell(v, f='.4f'):
    if v is None:
        return "-"
    try:
        return format(v, f)
    except Exception:
        return str(v)


def print_table(results, sort_by, result_dir):
    key = {
        'auc1':  lambda r: r.get('metrics', {}).get('AUC1', 0) or 0,
        'ap1':   lambda r: r.get('metrics', {}).get('AP1', 0) or 0,
        'infer': lambda r: -(get_infer_ms(r) or 1e9),
        'e2e':   lambda r: -(get_e2e_ms(r) or 1e9),
    }.get(sort_by, lambda r: r.get('metrics', {}).get('AUC1', 0) or 0)
    results = sorted(results, key=key, reverse=True)

    w = [20, 9, 9, 12, 12, 12, 20]
    headers = ['Model', 'AUC1', 'AP1', 'Infer ms', 'E2E ms', 'Extract fps', 'Timestamp']
    sep = '+' + '+'.join('-' * c for c in w) + '+'
    fmt = '|' + '|'.join(f'{{:^{c}}}' for c in w) + '|'

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in results:
        se = r.get("speed_end2end", {})
        print(fmt.format(
            str(r.get('model_name', '?'))[:w[0]-2],
            cell(r.get('metrics', {}).get('AUC1')),
            cell(r.get('metrics', {}).get('AP1')),
            cell(get_infer_ms(r), '.2f'),
            cell(get_e2e_ms(r), '.2f'),
            cell(se.get('image_throughput_fps'), '.1f'),
            str(r.get('timestamp', '?'))[:w[6]-2],
        ))
    print(sep)

    print(f"\n{len(results)} results  |  sorted by: {sort_by.upper()}  |  dir: {result_dir}\n")
    for r in results:
        info = r.get('model_info', {})
        print(f"  [{r.get('model_name')}]  "
              f"params={info.get('parameters', 0):,}  "
              f"extractor={info.get('feature_extractor', '?')}  "
              f"| {r.get('label_note', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sort', default='auc1',
                    choices=['auc1', 'ap1', 'infer', 'e2e'])
    ap.add_argument('--result-dir', default=DEFAULT_RESULT_DIR,
                    help='folder with result JSON files')
    args = ap.parse_args()

    results = load_results(args.result_dir)
    if not results:
        print(f"No result files in: {args.result_dir}\nRun 'python benchmark.py' first.")
        return
    print_table(results, args.sort, args.result_dir)


if __name__ == '__main__':
    main()
