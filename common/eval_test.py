import argparse, os, yaml, random
import numpy as np
import torch
from datasets import get_dataset
from models import DenoisingDiffusion
from utils.sampling import generalized_steps


class Args:
    resume = ""
    sampling_timesteps = 5
    image_folder = "results/images/"
    seed = 2024


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def eval_subset(ds, diffusion, config, args, name, n=100, batch_size=4):
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    errs = []
    skip = config.sampling.num_diffusion_timesteps // args.sampling_timesteps
    seq = range(0, config.sampling.num_diffusion_timesteps, skip)

    with torch.no_grad():
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            xs = torch.stack([ds[i][0] for i in batch_idx])
            plocs = torch.stack([torch.tensor(ds[i][2], dtype=torch.float32) for i in batch_idx])

            x = xs.to(config.device)
            x = 2 * x - 1.0
            x_cond = x[:, :config.data.time_step]
            x_gt = x[:, config.data.time_step:]
            ploc = (plocs / config.data.patch_size).to(config.device)

            noise = torch.randn_like(x_gt)
            denoise_img, _ = generalized_steps(noise, x_cond, seq, diffusion.model, diffusion.betas, ploc)

            mse = (denoise_img - x_gt).square().mean(dim=(1, 2, 3, 4))
            errs.extend(mse.cpu().numpy().tolist())

    errs = np.array(errs)
    print(f"[{name}] n={len(errs)}  평균 MSE={errs.mean():.6f}  표준편차={errs.std():.6f}")
    return errs


def main():
    with open("configs/military.yml") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "/workspace/data/real_military_dataset/ckpts/military_final.pth"

    args = Args()
    args.resume = ckpt_path

    train_ds, test_ds = get_dataset(args, config)

    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(ckpt_path, ema=True)
    diffusion.model.eval()

    normal_errs = eval_subset(train_ds, diffusion, config, args, "Training/Normal (정상)")
    val_errs = eval_subset(test_ds, diffusion, config, args, "Validation/All (이상 포함 추정)")

    print("\n=== 요약 ===")
    print(f"정상 데이터 평균 복원 오차: {normal_errs.mean():.6f}")
    print(f"검증 데이터 평균 복원 오차: {val_errs.mean():.6f}")


if __name__ == "__main__":
    main()
