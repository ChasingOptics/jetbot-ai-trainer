import os
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF

# === Helper Functions ===
def get_x(path):
    return (float(int(path[3:6])) - 50.0) / 50.0

def get_y(path):
    return (float(int(path[7:10])) - 50.0) / 50.0

# === Dataset Class with Deduplication ===
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.random_hflips = random_hflips

        # Collect all image paths, then dedupe
        paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            paths.extend(glob.glob(os.path.join(directory, ext)))
        self.image_paths = sorted(set(paths))

        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        x = get_x(os.path.basename(img_path))
        y = get_y(os.path.basename(img_path))

        if self.random_hflips and np.random.rand() > 0.5:
            image = TF.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()

if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR     = 'dataset_xy'
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 50
    TEST_FRAC    = 0.1
    NUM_WORKERS  = min(8, multiprocessing.cpu_count())
    USE_CUDA     = torch.cuda.is_available()
    DEVICE       = torch.device('cuda' if USE_CUDA else 'cpu')

    # --- Startup Info ---
    print(f"🖥️  Device: {DEVICE}")
    print(f"📦 Dataset directory: {DATA_DIR}")
    print(f"🔢 Num workers: {NUM_WORKERS}  |  Batch size: {BATCH_SIZE}")

    # --- Load Dataset ---
    dataset = XYDataset(DATA_DIR, random_hflips=True)
    print(f"🔍 Found {len(dataset)} images")

    # --- Warm-Up Load Test ---
    loader_test = DataLoader(dataset, batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    imgs, lbls = next(iter(loader_test))
    print(f"✅ Warm-up: loaded batch of {imgs.size(0)} images")

    # --- Split Dataset ---
    num_test = int(TEST_FRAC * len(dataset))
    train_ds, test_ds = random_split(dataset, [len(dataset) - num_test, num_test])

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA
    )

    # --- Model & Optimizer ---
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    best_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n🟢 Starting Epoch {epoch}/{NUM_EPOCHS}")
        start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            out = model(imgs)
            loss = F.mse_loss(out, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                lbls = lbls.to(DEVICE, non_blocking=True)
                out = model(imgs)
                test_loss += F.mse_loss(out, lbls).item()
        test_loss /= len(test_loader)

        duration = time.time() - start_time
        print(f"🔁 Epoch {epoch}/{NUM_EPOCHS} → Train: {train_loss:.4f}, Test: {test_loss:.4f}")
        print(f"⏱️ Epoch {epoch} took {duration:.1f}s")

        if test_loss < best_loss:
            torch.save(model.state_dict(), 'best_steering_model_xy.pth')
            best_loss = test_loss
            print("✅ Saved new best model")
