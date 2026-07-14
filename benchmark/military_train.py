import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os

import _env  # noqa: F401  — VadCLIP src를 sys.path에 추가
from model import CLIPVAD
from military_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import military_option

# 군 경계 데이터셋 라벨 맵
# key: CSV의 label 컬럼 값 / value: CLIP 텍스트 프롬프트
LABEL_MAP = {
    'Normal':    'normal maritime surveillance activity',
    'Approach':  'vessel approaching restricted military zone',
    'Intrusion': 'unauthorized vessel intrusion detected',
    'Threat':    'critical maritime security threat',
}


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp).view(1)], dim=0)
    return F.binary_cross_entropy(instance_logits, labels)


def train(model, normal_loader, anomaly_loader, test_loader, args, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(LABEL_MAP)
    ap_best = 0
    start_epoch = 0

    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        ap_best = ckpt['ap']
        print(f"checkpoint loaded: epoch={start_epoch+1}, ap={ap_best:.4f}")

    gt, gtsegments, gtlabels = None, None, None
    if os.path.exists(args.gt_path):
        gt = np.load(args.gt_path)
        gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
        gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    for e in range(start_epoch, args.max_epoch):
        model.train()
        loss_total1 = loss_total2 = 0
        normal_iter  = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            n_feat, n_label, n_len = next(normal_iter)
            a_feat, a_label, a_len = next(anomaly_iter)

            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            texts  = list(n_label) + list(a_label)
            lengths = torch.cat([n_len, a_len], dim=0).to(device)
            text_labels = get_batch_label(texts, prompt_text, LABEL_MAP).to(device)

            _, logits1, logits2 = model(visual, None, prompt_text, lengths)

            loss1 = CLAS2(logits1, text_labels, lengths, device)
            loss2 = CLASM(logits2, text_labels, lengths, device)
            loss_total1 += loss1.item()
            loss_total2 += loss2.item()

            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {e+1}/{args.max_epoch} | loss1={loss_total1/(i+1):.4f} | loss2={loss_total2/(i+1):.4f}")

        if gt is not None:
            AUC, AP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
            if AUC > ap_best:
                ap_best = AUC
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': ap_best,
                }, args.checkpoint_path)
                print(f"  checkpoint saved (AUC={ap_best:.4f})")

        torch.save(model.state_dict(), os.path.join(model_dir, 'model_military_cur.pth'))
        scheduler.step()

    torch.save(model.state_dict(), args.model_path)
    print(f"final model saved: {args.model_path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = military_option.parser.parse_args()
    setup_seed(args.seed)

    normal_dataset  = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, normal=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, normal=False)
    test_dataset    = UCFDataset(args.visual_length, args.test_list,  True,  LABEL_MAP)

    print(f"Normal seqs: {len(normal_dataset)}  |  Anomaly seqs: {len(anomaly_dataset)}  |  Test: {len(test_dataset)}")

    normal_loader  = DataLoader(normal_dataset,  batch_size=args.batch_size, shuffle=True,  drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    test_loader    = DataLoader(test_dataset,    batch_size=1,               shuffle=False)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, device)
