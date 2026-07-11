import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class MilitaryBoundaryDataset(Dataset):
    def __init__(self, data_root, transform=None, resize_shape=(256, 256), frame_length=5, patch_size=64):
        self.data_root = data_root
        self.transform = transform
        self.resize_height, self.resize_width = resize_shape
        self.frame_length = frame_length
        self.patch_size = patch_size
        
        # AI Hub 정상 데이터 타겟 경로
        normal_dir = os.path.join(data_root, 'Training', 'Normal') 
        self.video_files = []
        if os.path.exists(normal_dir):
            self.video_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(('.mp4', '.avi'))]
        print(f"[Dataset] Found {len(self.video_files)} videos in {normal_dir}")

    def __len__(self):
        return len(self.video_files) if len(self.video_files) > 0 else 100

    def __getitem__(self, idx):
        if len(self.video_files) == 0:
            dummy_patch = torch.rand(self.frame_length, 3, self.patch_size, self.patch_size)
            dummy_ploc = np.array([0, 0])
            return dummy_patch, "sample_video", dummy_ploc

        video_path = self.video_files[idx]
        video_name = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        while len(frames) < self.frame_length:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        cap.release()
        
        while len(frames) < self.frame_length:
            frames.append(torch.zeros(3, self.resize_height, self.resize_width))
            
        rgb_img = torch.stack(frames, dim=0)
        i_coord = np.random.randint(0, self.resize_height - self.patch_size + 1)
        j_coord = np.random.randint(0, self.resize_width - self.patch_size + 1)
        
        patch = rgb_img[:, :, i_coord:i_coord+self.patch_size, j_coord:j_coord+self.patch_size]
        ploc = np.array([i_coord, j_coord])
        
        return patch, video_name, ploc
