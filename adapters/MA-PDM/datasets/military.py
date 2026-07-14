import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


def np_load_frame(filename, resize_height, resize_width):
    image = cv2.imread(filename)
    image = image[:, :, ::-1]
    image = cv2.resize(image, (resize_width, resize_height))
    return image


class MilitaryBoundaryDataset(Dataset):
    def __init__(self, data_root, phase="train", transform=None,
                 resize_shape=(256, 256), frame_length=5, patch_size=64):
        self.data_root = data_root
        self.phase = phase
        self.transform = transform
        self.resize_height, self.resize_width = resize_shape
        self.frame_length = frame_length
        self.patch_size = patch_size

        if phase == "train":
            self.video_dir = os.path.join(data_root, "Training", "Normal")
        else:
            self.video_dir = os.path.join(data_root, "Validation", "All")

        self.videos = {}
        video_folders = sorted(glob.glob(os.path.join(self.video_dir, "*")))
        for vf in video_folders:
            name = os.path.basename(vf)
            frames = sorted(glob.glob(os.path.join(vf, "*.jpg")))
            if len(frames) >= self.frame_length:
                self.videos[name] = frames

        self.samples = []
        for name, frames in self.videos.items():
            for i in range(len(frames) - self.frame_length + 1):
                self.samples.append((name, i))

        print(f"[Dataset:{phase}] {len(self.videos)}개 클립, {len(self.samples)}개 샘플 ({self.video_dir})")

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 1

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            dummy_patch = torch.rand(self.frame_length, 3, self.patch_size, self.patch_size)
            return dummy_patch, "empty", np.array([0, 0])

        video_name, start = self.samples[idx]
        frame_paths = self.videos[video_name][start:start + self.frame_length]

        frames = []
        for fp in frame_paths:
            img = np_load_frame(fp, self.resize_height, self.resize_width)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img.copy()).permute(2, 0, 1).float() / 255.0
            frames.append(img)

        rgb_img = torch.stack(frames, dim=0)

        i_coord = np.random.randint(0, self.resize_height - self.patch_size + 1)
        j_coord = np.random.randint(0, self.resize_width - self.patch_size + 1)
        patch = rgb_img[:, :, i_coord:i_coord + self.patch_size, j_coord:j_coord + self.patch_size]
        ploc = np.array([i_coord, j_coord])

        return patch, video_name, ploc
