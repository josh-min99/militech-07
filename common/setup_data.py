import os, re, json, glob, shutil
from collections import defaultdict

SRC_ROOT = "/workspace/data/_extracted/51.군 경계 작전 환경 내 인식 데이터/3.개방데이터/1.데이터"
DST = "/workspace/data/real_military_dataset"
PATTERN = re.compile(r"^(.*)_(\d{4})(\d{3})\.jpg$")
ABNORMAL_CLASSES = {"3"}

def find_label_json(split, frame_path):
    frame_name = os.path.basename(frame_path)
    return os.path.join(SRC_ROOT, split, "02.라벨링데이터", frame_name.replace(".jpg", ".json"))

clips = defaultdict(list)
for split in ["Training", "Validation"]:
    frame_dir = os.path.join(SRC_ROOT, split, "01.원천데이터")
    for f in glob.glob(os.path.join(frame_dir, "*.jpg")):
        m = PATTERN.match(os.path.basename(f))
        if not m:
            continue
        prefix, clipid, frameidx = m.groups()
        clips[(split, prefix, clipid)].append((int(frameidx), f))

print(f"[setup_data] 총 클립 수: {len(clips)}")

n_normal, n_abnormal = 0, 0
for (split, prefix, clipid), frames in clips.items():
    frames.sort(key=lambda x: x[0])
    is_abnormal = False
    for idx, fpath in frames:
        label_path = find_label_json(split, fpath)
        if not os.path.exists(label_path):
            continue
        try:
            d = json.load(open(label_path))
            for a in d.get("annotations", []):
                if str(a.get("class")) in ABNORMAL_CLASSES:
                    is_abnormal = True
                    break
        except Exception:
            pass
        if is_abnormal:
            break

    clip_name = f"{prefix}_{clipid}"
    if is_abnormal:
        out_dir = os.path.join(DST, "Validation", "All", clip_name)
        n_abnormal += 1
    else:
        out_dir = os.path.join(DST, "Training", "Normal", clip_name)
        n_normal += 1

    os.makedirs(out_dir, exist_ok=True)
    for idx, fpath in frames:
        shutil.copy2(fpath, os.path.join(out_dir, os.path.basename(fpath)))

print(f"[setup_data] Training/Normal 클립 수: {n_normal}")
print(f"[setup_data] Validation/All   클립 수: {n_abnormal}")
