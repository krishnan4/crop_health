from icrawler.builtin import GoogleImageCrawler
import os

# Install first: pip install icrawler

diseases = [
    ("tomato late blight leaf real photo", "Tomato_Late_blight"),
    ("tomato early blight leaf photo",     "Tomato_Early_blight"),
    ("tomato healthy leaf green photo",    "Tomato_healthy"),
    ("potato blight leaf real photo",      "Potato___Early_blight"),
    ("potato late blight leaf photo",      "Potato___Late_blight"),
    ("pepper bacterial spot leaf photo",   "Pepper__bell___Bacterial_spot"),
    ("tomato leaf mold disease photo",     "Tomato_Leaf_Mold"),
    ("tomato yellow curl virus leaf",      "Tomato__Tomato_YellowLeaf__Curl_Virus"),
]

for keyword, folder_name in diseases:
    save_path = f"extra_images/{folder_name}"
    os.makedirs(save_path, exist_ok=True)
    print(f"Downloading: {keyword}")
    crawler = GoogleImageCrawler(storage={"root_dir": save_path})
    crawler.crawl(keyword=keyword, max_num=80)
    print(f"  ✅ Done — {folder_name}")

print("\n✅ All downloads complete!")