import os, shutil, random
from pathlib import Path

# ── CHANGE THESE TWO PATHS IF NEEDED ──
SRC = r"C:\Users\Admin\Desktop\crop_health\PlantVillage"
DST = r"C:\Users\Admin\Desktop\crop_health\plant_data"

random.seed(42)

classes = [f for f in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, f))]
print(f"Found {len(classes)} classes: {classes}")

for cls in classes:
    imgs = list(Path(SRC, cls).glob("*.*"))
    imgs = [i for i in imgs if i.suffix.lower() in [".jpg",".jpeg",".png",".JPG"]]
    random.shuffle(imgs)
    n = len(imgs)
    cuts = [int(n*0.70), int(n*0.85)]
    parts = [imgs[:cuts[0]], imgs[cuts[0]:cuts[1]], imgs[cuts[1]:]]
    splits = ["train", "val", "test"]
    for split, part in zip(splits, parts):
        out = Path(DST, split, cls)
        out.mkdir(parents=True, exist_ok=True)
        for img in part:
            shutil.copy(img, out / img.name)
    print(f"  {cls}: {len(parts[0])} train | {len(parts[1])} val | {len(parts[2])} test")

print("\n✅ Dataset ready!")