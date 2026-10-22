import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CUBDataset(Dataset):
    def __init__(self, data_root, samples, transform=None):
        self.data_root = Path(data_root)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(self.data_root / "images" / str(img_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def load_cub_metadata(data_root):
    root = Path(data_root)
    images = pd.read_csv(root / "images.txt", sep=r"\s+", names=["id", "img_path"], engine="python")
    labels = pd.read_csv(root / "image_class_labels.txt", sep=r"\s+", names=["id", "label"], engine="python")
    df = images.merge(labels, on="id").sort_values("id").reset_index(drop=True)
    df["label"] = df["label"].astype(int) - 1
    return df


def create_kfold_splits(data_root, n_splits=10, seed=42, save_dir="splits"):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    os.makedirs(save_dir, exist_ok=True)
    df = load_cub_metadata(data_root)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, df["label"])):
        df.iloc[train_idx].to_csv(Path(save_dir) / f"train_fold{fold}.csv", index=False)
        df.iloc[val_idx].to_csv(Path(save_dir) / f"val_fold{fold}.csv", index=False)
        print(f"Fold {fold + 1}/{n_splits}: {len(train_idx)} train / {len(val_idx)} val")


def get_split(data_root, fold_idx, transform=None, split_dir="splits"):
    train_df = pd.read_csv(Path(split_dir) / f"train_fold{fold_idx}.csv")
    val_df = pd.read_csv(Path(split_dir) / f"val_fold{fold_idx}.csv")
    train = CUBDataset(data_root, zip(train_df.img_path, train_df.label), transform)
    val = CUBDataset(data_root, zip(val_df.img_path, val_df.label), transform)
    return train, val


def get_dataloader(data_root, fold_idx, transform, batch_size=32, num_workers=0, split_dir="splits", val_transform=None):
    train_dataset, val_dataset = get_split(data_root, fold_idx, transform, split_dir)
    if val_transform is not None:
        _, val_dataset = get_split(data_root, fold_idx, val_transform, split_dir)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )
