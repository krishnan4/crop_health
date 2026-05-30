import torch, timm, os, shutil, random
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

GATE_DATA  = r"C:\Users\Admin\Desktop\crop_health\gate_data2"
OUT_MODEL  = r"C:\Users\Admin\Desktop\crop_health\leaf_gate.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Build better gate dataset ──
print("Building improved gate dataset...")

leaf_out = Path(GATE_DATA, "train", "leaf")
not_out  = Path(GATE_DATA, "train", "not_leaf")
leaf_out.mkdir(parents=True, exist_ok=True)
not_out.mkdir(parents=True, exist_ok=True)

# Collect leaf images from ALL plant classes
SRC_PLANT = r"C:\Users\Admin\Desktop\crop_health\PlantVillage"
leaf_imgs = []
for cls in os.listdir(SRC_PLANT):
    if cls == "Not_a_leaf":
        continue
    folder = Path(SRC_PLANT, cls)
    if not folder.is_dir():
        continue
    imgs = (list(folder.glob("*.jpg")) +
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.png")))
    leaf_imgs.extend(imgs[:80])  # 80 per class

random.shuffle(leaf_imgs)

# Copy leaf images WITH augmentation to simulate real-world
print(f"Processing {len(leaf_imgs)} leaf images with augmentation...")
for i, img_path in enumerate(leaf_imgs):
    try:
        img = Image.open(img_path).convert("RGB")

        # Save original
        img.save(leaf_out / f"leaf_{i}_orig.jpg")

        # Save darkened version (simulates bad lighting)
        dark = ImageEnhance.Brightness(img).enhance(0.5)
        dark.save(leaf_out / f"leaf_{i}_dark.jpg")

        # Save with color filter (simulates Instagram filters)
        filtered = ImageEnhance.Color(img).enhance(0.3)
        filtered.save(leaf_out / f"leaf_{i}_filtered.jpg")

        # Save rotated (simulates different angles)
        rotated = img.rotate(random.randint(15, 165))
        rotated.save(leaf_out / f"leaf_{i}_rotated.jpg")

        # Save with background (simulates soil/multiple leaves)
        # Add random colored border to simulate background
        bg_color = (
            random.randint(50, 150),
            random.randint(50, 120),
            random.randint(0, 80)
        )
        bg = Image.new("RGB", (img.width + 60, img.height + 60), bg_color)
        bg.paste(img, (30, 30))
        bg.save(leaf_out / f"leaf_{i}_withbg.jpg")

    except Exception as e:
        print(f"  Skip {img_path}: {e}")

# Copy not-leaf images
SRC_NOTLEAF = r"C:\Users\Admin\Desktop\crop_health\PlantVillage\Not_a_leaf"
not_imgs = (list(Path(SRC_NOTLEAF).glob("*.jpg")) +
            list(Path(SRC_NOTLEAF).glob("*.jpeg")) +
            list(Path(SRC_NOTLEAF).glob("*.png")))
random.shuffle(not_imgs)

# Match count with leaf images
leaf_count = len(list(leaf_out.glob("*.jpg")))
for i, img in enumerate(not_imgs[:leaf_count]):
    try:
        shutil.copy(img, not_out / f"notleaf_{i}{img.suffix}")
    except:
        pass

print(f"Leaf images: {len(list(leaf_out.glob('*')))}")
print(f"Not-leaf images: {len(list(not_out.glob('*')))}")

# ── Training ──
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.6, contrast=0.6,
                          saturation=0.5, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = datasets.ImageFolder(GATE_DATA + "/train", train_tf)
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
print(f"Classes: {ds.classes}")
print(f"Total training images: {len(ds)}")

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_acc = 0
for epoch in range(15):
    model.train()
    correct = total = 0
    for imgs, labels in tqdm(dl, desc=f"Gate Epoch {epoch+1}/15"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}: acc={acc:.1%}")
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state': model.state_dict(),
            'classes': ds.classes
        }, OUT_MODEL)
        print(f"  ✅ Saved best gate model ({acc:.1%})")

print(f"\n🎉 Gate training complete! Best accuracy: {best_acc:.1%}")
print("leaf_gate.pth updated!")