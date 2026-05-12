import os
from pathlib import Path
import time
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class ManifestDataset(Dataset):
    def __init__(self, df, class_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row['class']]
        return img, label


def get_transforms(img_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_dataloaders(manifest_path, batch_size=16, img_size=224, num_workers=2):
    df = pd.read_csv(manifest_path)
    classes = sorted(df['class'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    dfs = {s: df[df['split'] == s].reset_index(drop=True) for s in ['train', 'val', 'test']}

    ds_train = ManifestDataset(dfs['train'], class_to_idx, transform=get_transforms(img_size, train=True))
    ds_val = ManifestDataset(dfs['val'], class_to_idx, transform=get_transforms(img_size, train=False))
    ds_test = ManifestDataset(dfs['test'], class_to_idx, transform=get_transforms(img_size, train=False))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return dl_train, dl_val, dl_test, class_to_idx


def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def evaluate(model, device, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main(manifest_path='data/split_manifest.csv', batch_size=16, img_size=224, epochs=1, lr=1e-3, out_dir='output/04_model'):
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val, dl_test, class_to_idx = get_dataloaders(manifest_path, batch_size=batch_size, img_size=img_size)

    from src.models import get_efficientnet_b0
    model = get_efficientnet_b0(num_classes=len(class_to_idx), pretrained=True)
    model = model.to(device)

    # compute class weights
    import pandas as pd
    df = pd.read_csv(manifest_path)
    counts = df[df['split'] == 'train']['class'].value_counts().sort_index()
    # ensure order matches class_to_idx
    counts = counts.reindex(sorted(counts.index))
    freq = counts.values.astype(float)
    class_weights = torch.tensor(1.0 / (freq + 1e-12), dtype=torch.float32).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    ckpt_path = Path('checkpoints')
    ckpt_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, device, dl_train, criterion, optimizer)
        val_loss, val_acc = evaluate(model, device, dl_val, criterion)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'class_to_idx': class_to_idx}, ckpt_path / 'best_model.pth')
            print(f"  ✅ Saved best checkpoint: {ckpt_path / 'best_model.pth'}")

    # final test eval
    test_loss, test_acc = evaluate(model, device, dl_test, criterion)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == '__main__':
    main(epochs=1)
