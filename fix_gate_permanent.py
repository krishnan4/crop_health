import torch, timm, os, shutil, random
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

PLANTVILLAGE = r"C:\Users\Admin\Desktop\crop_health\PlantVillage"
NOT_LEAF_SRC = r"C:\Users\Admin\Desktop\crop_health\PlantVillage\Not_a_leaf"
GATE_DATA    = r"C:\Users\Admin\Desktop\crop_health\gate_data_v2"
OUT_MODEL    = r"C:\Users\Admin\Desktop\crop_health\leaf_gate.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 5     # reduced from 12
BATCH        = 32    # increased from 16 — faster per epoch
MAX_LEAF     = 20    # only 20 images per class (was 60) — 3x fewer images

print("PERMANENT GATE MODEL FIX (Fast Version)")
print("Device:", DEVICE)

leaf_out = Path(GATE_DATA, "train", "leaf")
not_out  = Path(GATE_DATA, "train", "not_leaf")
leaf_out.mkdir(parents=True, exist_ok=True)
not_out.mkdir(parents=True, exist_ok=True)

for f in leaf_out.glob("*"): f.unlink()
for f in not_out.glob("*"):  f.unlink()

def augment_and_save(img, out_dir, prefix, idx):
    w, h = img.size
    # Only 5 versions instead of 13 — covers all key real-world conditions
    img.save(out_dir / f"{prefix}_{idx}_orig.jpg")
    ImageEnhance.Brightness(img).enhance(0.3).save(out_dir / f"{prefix}_{idx}_dark.jpg")
    ImageEnhance.Brightness(img).enhance(1.8).save(out_dir / f"{prefix}_{idx}_bright.jpg")
    img.convert("L").convert("RGB").save(out_dir / f"{prefix}_{idx}_grey.jpg")

    bg_colors = [(101,67,33),(34,85,34),(180,160,120)]
    pad = 40
    bg = Image.new("RGB", (w + pad*2, h + pad*2), random.choice(bg_colors))
    bg.paste(img, (pad, pad))
    bg.save(out_dir / f"{prefix}_{idx}_withbg.jpg")

print("\n[1/4] Collecting leaf images...")
leaf_imgs = []
for cls in os.listdir(PLANTVILLAGE):
    if cls.lower() in ("not_a_leaf", "plantvillage"):
        continue
    folder = Path(PLANTVILLAGE, cls)
    if not folder.is_dir():
        continue
    imgs = (list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) +
            list(folder.glob("*.png")) + list(folder.glob("*.jpeg")))
    random.shuffle(imgs)
    leaf_imgs.extend(imgs[:MAX_LEAF])

random.shuffle(leaf_imgs)
print(f"Found {len(leaf_imgs)} source leaf images")

print("[2/4] Augmenting and saving...")
for idx, img_path in enumerate(leaf_imgs):
    try:
        img = Image.open(img_path).convert("RGB")
        augment_and_save(img, leaf_out, "leaf", idx)
    except:
        pass

total_leaf = len(list(leaf_out.glob("*")))
print(f"Created {total_leaf} leaf images")

not_imgs = []
for ext in ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]:
    not_imgs.extend(Path(NOT_LEAF_SRC).glob(ext))
random.shuffle(not_imgs)
for idx, img_path in enumerate(not_imgs[:total_leaf]):
    try:
        shutil.copy(img_path, not_out / f"notleaf_{idx}{img_path.suffix}")
    except:
        pass
print(f"Created {len(list(not_out.glob('*')))} not-leaf images")

print(f"\n[3/4] Training gate model ({EPOCHS} epochs, batch={BATCH})...")
print("Estimated time: 15-25 minutes on CPU")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

ds = datasets.ImageFolder(GATE_DATA + "/train", train_tf)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
print(f"Classes: {ds.classes} | Total images: {len(ds)}")

model     = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model     = model.to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    correct = total = 0
    total_loss = 0
    for imgs, labels in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item()
    acc = correct / total
    scheduler.step()
    print(f"Epoch {epoch+1}: loss={total_loss/len(dl):.3f} | acc={acc:.1%}")
    if acc > best_acc:
        best_acc = acc
        torch.save({'model_state': model.state_dict(), 'classes': ds.classes}, OUT_MODEL)
        print(f"  Saved best model ({acc:.1%})")

print(f"\n[4/4] Done! Best accuracy: {best_acc:.1%}")
print("leaf_gate.pth updated!")
print("\nNow restart your app:")
print("uvicorn main:app --reload --host 127.0.0.1 --port 8000")