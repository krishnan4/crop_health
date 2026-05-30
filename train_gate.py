import torch, timm, os, shutil, random
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Build binary dataset: leaf vs not_leaf
SRC_PLANT   = r"C:\Users\Admin\Desktop\crop_health\PlantVillage"
SRC_NOTLEAF = r"C:\Users\Admin\Desktop\crop_health\PlantVillage\Not_a_leaf"
GATE_DATA   = r"C:\Users\Admin\Desktop\crop_health\gate_data"
OUT_MODEL   = r"C:\Users\Admin\Desktop\crop_health\leaf_gate.pth"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print("Building gate dataset...")

# Collect leaf images (500 from each plant class)
leaf_out = Path(GATE_DATA, "train", "leaf")
leaf_out.mkdir(parents=True, exist_ok=True)
not_out  = Path(GATE_DATA, "train", "not_leaf")
not_out.mkdir(parents=True, exist_ok=True)

leaf_imgs = []
for cls in os.listdir(SRC_PLANT):
    if cls == "Not_a_leaf":
        continue
    folder = Path(SRC_PLANT, cls)
    if not folder.is_dir():
        continue
    imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.png"))
    leaf_imgs.extend(imgs[:50])  # 50 per class

random.shuffle(leaf_imgs)
for i, img in enumerate(leaf_imgs):
    shutil.copy(img, leaf_out / f"leaf_{i}{img.suffix}")

# Copy not_leaf images
not_imgs = list(Path(SRC_NOTLEAF).glob("*.*"))
not_imgs = [i for i in not_imgs if i.suffix.lower() in [".jpg",".jpeg",".png"]]
random.shuffle(not_imgs)
for i, img in enumerate(not_imgs[:len(leaf_imgs)]):
    shutil.copy(img, not_out / f"notleaf_{i}{img.suffix}")

print(f"Leaf images: {len(leaf_imgs)} | Not-leaf images: {min(len(not_imgs), len(leaf_imgs))}")

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ds = datasets.ImageFolder(GATE_DATA+"/train", tf)
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
print(f"Classes: {ds.classes}")

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    correct = total = 0
    for imgs, labels in tqdm(dl, desc=f"Gate Epoch {epoch+1}/10"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    print(f"Epoch {epoch+1}: acc={correct/total:.1%}")

torch.save({'model_state': model.state_dict(), 'classes': ds.classes}, OUT_MODEL)
print("✅ Gate model saved as leaf_gate.pth")