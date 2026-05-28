import torch, timm, json
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR  = r"C:\Users\Admin\Desktop\crop_health\plant_data"
OUT_MODEL = r"C:\Users\Admin\Desktop\crop_health\best_model.pth"
EPOCHS    = 20
BATCH     = 16
LR        = 1e-4
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(DATA_DIR+"/train", train_tf)
val_ds   = datasets.ImageFolder(DATA_DIR+"/val",   val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

NUM_CLASSES = len(train_ds.classes)
print(f"Total classes: {NUM_CLASSES}")
print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

# Save class names immediately
with open(r"C:\Users\Admin\Desktop\crop_health\class_names.json", "w") as f:
    json.dump(train_ds.classes, f)
print("✅ Saved class_names.json")

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc = correct / total
    scheduler.step()
    print(f"\n📊 Epoch {epoch+1}: loss={total_loss/len(train_dl):.3f} | val_acc={acc:.1%}")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state': model.state_dict(),
            'classes': train_ds.classes,
            'num_classes': NUM_CLASSES
        }, OUT_MODEL)
        print(f"  ✅ New best model saved! ({acc:.1%})")

print(f"\n🎉 Training complete! Best accuracy: {best_acc:.1%}")