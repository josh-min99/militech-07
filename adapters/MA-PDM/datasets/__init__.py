import os
from .military import MilitaryBoundaryDataset

def get_dataset(args, config):
    dataset_name = config.data.dataset
    data_root = config.data.data_dir

    if dataset_name.lower() == 'military':
        train_dataset = MilitaryBoundaryDataset(
            data_root=data_root,
            phase="train",
            resize_shape=(config.data.image_size, config.data.image_size),
            frame_length=config.data.time_step + 1,
            patch_size=config.data.patch_size
        )
        test_dataset = MilitaryBoundaryDataset(
            data_root=data_root,
            phase="test",
            resize_shape=(config.data.image_size, config.data.image_size),
            frame_length=config.data.time_step + 1,
            patch_size=config.data.patch_size
        )
        return train_dataset, test_dataset
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")
