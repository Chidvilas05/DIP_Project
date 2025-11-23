


import os
import time
import copy
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import torchvision
import numpy as np


class PreprocessedImageFolder(Dataset):
    def __init__(self, root, transform=None, preprocess_fn=None):
        self.root = root
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.base = datasets.ImageFolder(root)
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.samples = self.base.samples  

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.preprocess_fn(path) 

        if self.transform:
            img = self.transform(img)

        return img, label


DATA_DIR = "./transformed_stanford_cars_short/"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224
BACKBONE = "resnet50"
FREEZE_BACKBONE = False


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (mps)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, train_dataset, val_dataset, val_transforms




class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(num_classes, backbone=None, freeze_backbone=False):
    model = SmallResNet(num_classes)
    return model


def train_model(model, dataloaders, device, num_epochs=20, lr=1e-4, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        elapsed = time.time() - start_time
        print(f"Epoch time: {elapsed:.1f} sec")
        torch.save(model.state_dict(), f"intermediate_model_dup1_short.pth")

    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)
    acc = correct.float() / total
    print(f"Test accuracy: {acc:.4f}")
    return acc


def predict_single_image(model, img_path, device, transform, class_names, topk=5):
    model.eval()
    model.to(device)

    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=topk)
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    print(f"\nTop-{topk} predictions for {img_path}:")
    for p, idx in zip(top_probs, top_idxs):
        print(f"{class_names[idx]}: {p:.3f}")


def main():
    device = get_device()

    print("Creating dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset, val_transform = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4
    )

    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes.")

    print(f"Building model with backbone: {BACKBONE}")
    model = build_model(
        num_classes=num_classes,
        backbone=BACKBONE,
        freeze_backbone=FREEZE_BACKBONE
    )

    dataloaders = {"train": train_loader, "val": val_loader}

    print("Starting training...")
    model, history = train_model(
        model,
        dataloaders,
        device,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("\nEvaluating on test set...")
    evaluate_model(model, val_loader, device)

    out_path = Path("stanford_cars_{}_best.pth".format(BACKBONE))
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "backbone": BACKBONE,
        "freeze_backbone": FREEZE_BACKBONE,
        "img_size": IMG_SIZE,
    }, out_path)
    print(f"Saved best model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
