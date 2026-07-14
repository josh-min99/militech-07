import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import _env  # noqa: F401  — VadCLIP src를 sys.path에 추가
from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
import military_option

LABEL_MAP = {
    'Normal':    'normal maritime surveillance activity',
    'Approach':  'vessel approaching restricted military zone',
    'Intrusion': 'unauthorized vessel intrusion detected',
    'Threat':    'critical maritime security threat',
}


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()
    ap1_all, ap2_all = [], []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            rem = length
            for j in range(len(lengths)):
                lengths[j] = min(rem, maxlen)
                rem -= maxlen
                if rem <= 0:
                    break
            lengths = lengths.to(int)

            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)

            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])

            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            prob2 = 1 - logits2[0:len_cur].softmax(dim=-1)[:, 0]

            ap1_all.append(prob1.cpu().numpy())
            ap2_all.append(prob2.cpu().numpy())

    ap1 = np.concatenate(ap1_all)
    ap2 = np.concatenate(ap2_all)

    try:
        ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
        AP1  = average_precision_score(gt, np.repeat(ap1, 16))
        ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
        AP2  = average_precision_score(gt, np.repeat(ap2, 16))
        print(f"AUC1={ROC1:.4f}  AP1={AP1:.4f}")
        print(f"AUC2={ROC2:.4f}  AP2={AP2:.4f}")
        return ROC1, AP1
    except Exception as e:
        print(f"eval skipped (gt missing or shape mismatch): {e}")
        return 0.0, 0.0


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = military_option.parser.parse_args()

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    prompt_text  = get_prompt_text(LABEL_MAP)

    gt = np.load(args.gt_path) if hasattr(args, 'gt_path') and __import__('os').path.exists(args.gt_path) else None
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True) if gt is not None else None
    gtlabels   = np.load(args.gt_label_path,   allow_pickle=True) if gt is not None else None

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )
    model.load_state_dict(torch.load(args.model_path))

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
