import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from data import create_kfold_splits, get_dataloader
from model import CNN


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            total_correct += outputs.argmax(1).eq(labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(data_root="data/CUB_200_2011", epochs=10, batch_size=32, lr=1e-3,
          n_splits=10, seed=42, num_workers=0, patience=3, output_dir="outputs"):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = Path(output_dir) / "splits"
    create_kfold_splits(data_root, n_splits=n_splits, seed=seed, save_dir=split_dir)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    results = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(n_splits):
        print(f"\n===== Fold {fold + 1}/{n_splits} =====")
        train_loader, val_loader = get_dataloader(
            data_root, fold, train_transform, batch_size, num_workers, split_dir, val_transform
        )
        model = CNN(num_classes=200).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.3)
        best_acc, best_state, stale = -1.0, None, 0
        history = []
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step(val_acc)
            row = {"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc}
            history.append(row)
            print(f"Epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc, best_state, stale = val_acc, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    print("Early stopping")
                    break
        torch.save(best_state, output_dir / f"best_model_fold{fold}.pt")
        results.append({"fold": fold, "best_val_acc": best_acc, "history": history})
    mean_acc = float(np.mean([item["best_val_acc"] for item in results]))
    std_acc = float(np.std([item["best_val_acc"] for item in results]))
    summary = {"folds": results, "mean_best_val_acc": mean_acc, "std_best_val_acc": std_acc}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n10-fold mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CUB CNN with 10-fold cross-validation")
    parser.add_argument("--data-root", default="data/CUB_200_2011")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    train(data_root=args.data_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          n_splits=args.folds, seed=args.seed, num_workers=args.num_workers,
          patience=args.patience, output_dir=args.output_dir)
