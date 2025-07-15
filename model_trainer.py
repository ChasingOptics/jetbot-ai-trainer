# model_trainer.py

import os
import glob
import time
import numpy as np
import warnings
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import torch_directml

# === Suppress Deprecation Warnings ===
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === Set device ===
device = torch_directml.device()
print(f"🧠 Using DirectML device: {device}")

# === Helper functions ===
def get_x(path):
    return (float(int(path[3:6])) - 50.0) / 50.0

def get_y(path):
    return (float(int(path[7:10])) - 50.0) / 50.0

# === Custom dataset ===
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))

        if self.random_hflips and np.random.rand() > 0.5:
            image = TF.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()

# === Load dataset ===
dataset = XYDataset('dataset_xy', random_hflips=True)
print(f"📂 Found {len(dataset)} images")

# === Split dataset ===
test_percent = 0.1
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test])

# === DataLoaders ===
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=0)

# === Model ===
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# === Training settings ===
NUM_EPOCHS = 100
BEST_MODEL_PATH = 'best_steering_model_xy.pth'
best_loss = 1e9
no_improve_epochs = 0
early_stop_patience = 10

optimizer = optim.Adam(model.parameters())

# === Training loop ===
for epoch in range(NUM_EPOCHS):
    print(f"\n🟢 Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
    epoch_start = time.time()

    # Train
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = F.mse_loss(outputs, labels)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    # Time reporting
    epoch_time = time.time() - epoch_start
    print(f"📉 Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | ⏱️ Time: {epoch_time:.1f}s")

    if test_loss < best_loss:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_loss = test_loss
        no_improve_epochs = 0
        print("✅ Best model saved.")
    else:
        no_improve_epochs += 1
        print(f"⏸️ No improvement for {no_improve_epochs} epochs")

    if no_improve_epochs >= early_stop_patience:
        print("🛑 Early stopping triggered.")
        break
