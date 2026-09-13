import os
import shutil
import hashlib
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

RAW_DIR    = r"E:\VS Code\Projects\Thesis\TP002\PlantF\data_raw"
OUTPUT_DIR = r"E:\VS Code\Projects\Thesis\TP002\PlantF\Data"

RANDOM_SEED      = 42
TARGET_PER_CLASS = 500
TEST_RATIO       = 0.15
VAL_RATIO        = 0.15
MIN_IMAGES       = 50

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SEEN_CLASSES = [
    "Apple___Apple_Scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Corn___Common_rust", "Corn___Gray_leaf_spot", "Corn___Northern_Leaf_Blight", "Corn___Healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Mosaic_Virus", "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot", "Tomato___Two_Spotted_Spider_Mites", "Tomato___Yellow_Leaf_Curl_Virus",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___Healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___Healthy",
    "Strawberry___Healthy",
]

UNSEEN_CLASSES = [
    "Orange___Citrus_greening",
    "Peach___Healthy", "Peach___Bacterial_spot",
    "Blueberry___Healthy",
    "Raspberry___Healthy",
    "Squash___Powdery_mildew",
    "Cherry___Healthy", "Cherry___Powdery_mildew",
    "Soybean___Healthy",
    "Strawberry___Leaf_scorch",
    "Cassava___Brown_Streak_Disease", "Cassava___Green_Mottle_Disease",
]

RESERVED_CLASSES = [
    "Cassava___Bacterial_Blight",
    "Cassava___Healthy",
    "Cassava___Mosaic_Disease",
    "Pepper_Bell___Bacterial_spot",
    "Pepper_Bell___Healthy",
]

ALL_TARGET_CLASSES = SEEN_CLASSES + UNSEEN_CLASSES + RESERVED_CLASSES

COUNTRY_CLIENTS = {
    "Bangladesh": ["Potato", "Tomato"],
    "India":      ["Tomato", "Grape", "Apple"],
    "USA":        ["Corn", "Apple", "Grape", "Potato"],
    "Spain":      ["Grape"],
}

RAW_TO_UNIFIED = {
    "Apple___Apple_scab":                            "Apple___Apple_Scab",
    "Apple___Black_rot":                             "Apple___Black_rot",
    "Apple___Cedar_apple_rust":                      "Apple___Cedar_apple_rust",
    "Apple___healthy":                               "Apple___Healthy",
    "Blueberry___healthy":                           "Blueberry___Healthy",
    "Cherry___healthy":                              "Cherry___Healthy",
    "Cherry___Powdery_mildew":                       "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot":    "Corn___Gray_leaf_spot",
    "Corn___Common_rust":                            "Corn___Common_rust",
    "Corn___healthy":                                "Corn___Healthy",
    "Corn___Northern_Leaf_Blight":                   "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot":                             "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)":                  "Grape___Esca",
    "Grape___healthy":                               "Grape___Healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":    "Grape___Leaf_blight",
    "Orange___Haunglongbing_(Citrus_greening)":      "Orange___Citrus_greening",
    "Peach___Bacterial_spot":                        "Peach___Bacterial_spot",
    "Peach___healthy":                               "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot":                 "Pepper_Bell___Bacterial_spot",
    "Pepper,_bell___healthy":                        "Pepper_Bell___Healthy",
    "Potato___Early_blight":                         "Potato___Early_blight",
    "Potato___healthy":                              "Potato___Healthy",
    "Potato___Late_blight":                          "Potato___Late_blight",
    "Raspberry___healthy":                           "Raspberry___Healthy",
    "Soybean___healthy":                             "Soybean___Healthy",
    "Squash___Powdery_mildew":                       "Squash___Powdery_mildew",
    "Strawberry___healthy":                          "Strawberry___Healthy",
    "Strawberry___Leaf_scorch":                      "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot":                       "Tomato___Bacterial_spot",
    "Tomato___Early_blight":                         "Tomato___Early_blight",
    "Tomato___healthy":                              "Tomato___Healthy",
    "Tomato___Late_blight":                          "Tomato___Late_blight",
    "Tomato___Leaf_Mold":                            "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot":                   "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Two_Spotted_Spider_Mites",
    "Tomato___Target_Spot":                          "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus":                  "Tomato___Mosaic_Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":        "Tomato___Yellow_Leaf_Curl_Virus",
    "Cassava Bacterial Blight (CBB)":                "Cassava___Bacterial_Blight",
    "Cassava Brown Streak Disease (CBSD)":           "Cassava___Brown_Streak_Disease",
    "Cassava Green Mottle (CGM)":                    "Cassava___Green_Mottle_Disease",
    "Cassava Mosaic Disease (CMD)":                  "Cassava___Mosaic_Disease",
    "Healthy":                                       "Cassava___Healthy",
    "Apple Scab Leaf":                               "Apple___Apple_Scab",
    "Apple rust leaf":                               "Apple___Cedar_apple_rust",
    "Apple leaf":                                    "Apple___Healthy",
    "Blueberry leaf":                                "Blueberry___Healthy",
    "Cherry leaf":                                   "Cherry___Healthy",
    "Cherry Powdery Mildew":                         "Cherry___Powdery_mildew",
    "Corn Gray leaf spot":                           "Corn___Gray_leaf_spot",
    "Corn leaf blight":                              "Corn___Northern_Leaf_Blight",
    "Corn rust leaf":                                "Corn___Common_rust",
    "grape leaf":                                    "Grape___Healthy",
    "grape leaf black rot":                          "Grape___Black_rot",
    "Peach leaf":                                    "Peach___Healthy",
    "Potato leaf early blight":                      "Potato___Early_blight",
    "Potato leaf late blight":                       "Potato___Late_blight",
    "Raspberry leaf":                                "Raspberry___Healthy",
    "Soyabean leaf":                                 "Soybean___Healthy",
    "Squash Powdery mildew leaf":                    "Squash___Powdery_mildew",
    "Strawberry leaf":                               "Strawberry___Healthy",
    "Tomato Early blight leaf":                      "Tomato___Early_blight",
    "Tomato leaf":                                   "Tomato___Healthy",
    "Tomato leaf bacterial spot":                    "Tomato___Bacterial_spot",
    "Tomato leaf late blight":                       "Tomato___Late_blight",
    "Tomato leaf mosaic virus":                      "Tomato___Mosaic_Virus",
    "Tomato leaf yellow virus":                      "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":                              "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot":                     "Tomato___Septoria_leaf_spot",
    "Bell_pepper leaf":                              "Pepper_Bell___Healthy",
    "Bell_pepper leaf spot":                         "Pepper_Bell___Bacterial_spot",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_valid(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_dir(p):
    os.makedirs(p, exist_ok=True)


def copy_img(src, dst_dir, fname=None):
    fname = fname or os.path.basename(src)
    name  = os.path.splitext(fname)[0] + ".jpg"
    dst   = os.path.join(dst_dir, name)
    try:
        img = Image.open(src).convert("RGB")
        img.save(dst, "JPEG", quality=95)
        return name
    except Exception as e:
        print(f"  Warning: failed to copy {src}: {e}")
        return None


print("\n[Step 1] Collecting images from raw datasets...")

raw_class_images = defaultdict(list)

datasets = {
    "plantvillage": os.path.join(RAW_DIR, "plantvillage"),
    "plantdoc":     os.path.join(RAW_DIR, "plantdoc"),
    "cassava":      os.path.join(RAW_DIR, "cassava"),
}

for ds_name, ds_path in datasets.items():
    if not os.path.exists(ds_path):
        print(f"  Warning: {ds_path} not found, skipping")
        continue
    for root, dirs, files in os.walk(ds_path):
        folder_name = os.path.basename(root)
        unified     = RAW_TO_UNIFIED.get(folder_name)
        if unified is None:
            continue
        imgs = [os.path.join(root, f) for f in files
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if imgs:
            raw_class_images[unified].extend(imgs)
            print(f"  [{ds_name}] {folder_name} -> {unified}: {len(imgs)} images")

print(f"\n  Classes found: {len(raw_class_images)}")
for cls in sorted(raw_class_images):
    print(f"    {cls:<50} {len(raw_class_images[cls]):>5} images")

print("\n[Step 2] Deduplication and corrupt image removal...")

hash_registry   = {}
removed_corrupt = 0
removed_dup     = 0
clean_images    = defaultdict(list)

for cls in tqdm(sorted(raw_class_images), desc="  Cleaning"):
    for path in raw_class_images[cls]:
        if not is_valid(path):
            removed_corrupt += 1
            continue
        h = md5(path)
        if h in hash_registry:
            removed_dup += 1
            continue
        hash_registry[h] = path
        clean_images[cls].append(path)

print(f"  Corrupt images removed : {removed_corrupt}")
print(f"  Duplicate images removed: {removed_dup}")
for cls in sorted(clean_images):
    if cls in ALL_TARGET_CLASSES:
        print(f"  {cls:<50} {len(clean_images[cls]):>5} clean images")

print(f"\n[Step 3] Class-balanced sampling (cap={TARGET_PER_CLASS} per seen class)...")

balanced_images = {}
balance_report  = []

for cls in ALL_TARGET_CLASSES:
    images = clean_images.get(cls, [])
    n_raw  = len(images)

    if n_raw < MIN_IMAGES:
        print(f"  Skipping {cls}: only {n_raw} images (minimum={MIN_IMAGES})")
        balance_report.append({"class": cls, "raw": n_raw, "kept": 0, "note": "skipped"})
        continue

    if cls in SEEN_CLASSES:
        if n_raw > TARGET_PER_CLASS:
            kept = random.sample(images, TARGET_PER_CLASS)
            note = f"downsampled {n_raw} to {TARGET_PER_CLASS}"
        else:
            kept = images
            note = f"all {n_raw} kept"
    else:
        kept = images
        note = f"all {n_raw} kept"

    balanced_images[cls] = kept
    balance_report.append({"class": cls, "raw": n_raw, "kept": len(kept), "note": note})
    print(f"  {cls:<50} {n_raw:>5} -> {len(kept):>5}  ({note})")

# NOTE: balance_report.csv is saved further down, AFTER the output directory
# is (re)built in Step 4. Saving it here would be pointless: Step 4 calls
# shutil.rmtree(OUTPUT_DIR) if it already exists, which would immediately
# delete this file the moment it's written.

seen_sizes   = [len(balanced_images[c]) for c in SEEN_CLASSES if c in balanced_images]
unseen_sizes = [len(balanced_images[c]) for c in UNSEEN_CLASSES if c in balanced_images]
print(f"\n  Seen classes   — min:{min(seen_sizes)}  max:{max(seen_sizes)}  "
      f"mean:{np.mean(seen_sizes):.0f}  total:{sum(seen_sizes)}")
print(f"  Unseen classes — min:{min(unseen_sizes)}  max:{max(unseen_sizes)}  "
      f"total:{sum(unseen_sizes)}")

print(f"\n[Step 4] Building output directory structure at {OUTPUT_DIR}...")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    make_dir(os.path.join(OUTPUT_DIR, "global", split))
make_dir(os.path.join(OUTPUT_DIR, "zero_shot", "unseen_test"))
make_dir(os.path.join(OUTPUT_DIR, "reserved"))
make_dir(os.path.join(OUTPUT_DIR, "metadata"))
for client in COUNTRY_CLIENTS:
    for split in ["train", "val", "test"]:
        make_dir(os.path.join(OUTPUT_DIR, "federated", client, split))

# Saved here (not in Step 3) so it survives the rmtree/rebuild above.
pd.DataFrame(balance_report).to_csv(
    os.path.join(OUTPUT_DIR, "metadata", "balance_report.csv"), index=False)

print("\n[Step 5] Splitting and copying images...")

metadata    = []
class_to_id = {c: i for i, c in enumerate(ALL_TARGET_CLASSES)}

for cls in tqdm(ALL_TARGET_CLASSES, desc="  Processing classes"):
    if cls not in balanced_images:
        continue

    images = balanced_images[cls]
    crop   = cls.split("___")[0]

    if cls in RESERVED_CLASSES:
        dst_dir = os.path.join(OUTPUT_DIR, "reserved", cls)
        make_dir(dst_dir)
        for i, img in enumerate(images):
            copy_img(img, dst_dir, f"{cls}_{i:05d}.jpg")
        continue

    if cls in UNSEEN_CLASSES:
        dst_dir = os.path.join(OUTPUT_DIR, "zero_shot", "unseen_test", cls)
        make_dir(dst_dir)
        for i, img in enumerate(images):
            fname = copy_img(img, dst_dir, f"{cls}_{i:05d}.jpg")
            if fname:
                metadata.append({
                    "image": fname, "class": cls,
                    "class_id": class_to_id[cls], "crop": crop,
                    "type": "unseen", "split": "unseen_test",
                    "dataset": "mixed",
                })
        continue

    train_val, test = train_test_split(
        images, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    train, val = train_test_split(
        train_val, test_size=VAL_RATIO / (1 - TEST_RATIO), random_state=RANDOM_SEED)

    split_map = {"train": train, "val": val, "test": test}

    for split_name, img_list in split_map.items():
        dst = os.path.join(OUTPUT_DIR, "global", split_name, cls)
        make_dir(dst)
        for i, img in enumerate(img_list):
            fname = copy_img(img, dst, f"{cls}_{split_name}_{i:05d}.jpg")
            if fname:
                metadata.append({
                    "image": fname, "class": cls,
                    "class_id": class_to_id[cls], "crop": crop,
                    "type": "seen", "split": split_name,
                    "dataset": "mixed",
                })

    eligible = [c for c, crops in COUNTRY_CLIENTS.items() if crop in crops]
    if eligible:
        for split_name, img_list in split_map.items():
            parts = np.array_split(img_list, len(eligible))
            for client, part in zip(eligible, parts):
                dst = os.path.join(OUTPUT_DIR, "federated", client, split_name, cls)
                make_dir(dst)
                for i, img in enumerate(part):
                    copy_img(img, dst, f"{cls}_{client}_{split_name}_{i:05d}.jpg")

print("\n[Step 6] Saving metadata...")

df = pd.DataFrame(metadata)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata", "complete_metadata.csv"), index=False)
with open(os.path.join(OUTPUT_DIR, "metadata", "class_mapping.json"), "w") as f:
    json.dump(class_to_id, f, indent=2)

seen_df      = df[df["type"] == "seen"]
class_counts = seen_df["class"].value_counts().to_dict()
total        = sum(class_counts.values())
n_cls        = len(class_counts)
class_weights = {c: total / (n_cls * count) for c, count in class_counts.items()}
with open(os.path.join(OUTPUT_DIR, "metadata", "class_weights.json"), "w") as f:
    json.dump(class_weights, f, indent=2)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETE")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")

split_counts = df.groupby(["type", "split"]).size()
print(f"\nSplit image counts:")
print(split_counts.to_string())

print(f"\nSeen class sizes (after balancing):")
seen_final = df[df["type"] == "seen"].groupby("class").size().sort_values()
print(f"  Min: {seen_final.min()}  Max: {seen_final.max()}  "
      f"Std: {seen_final.std():.1f}  Total: {seen_final.sum()}")

print(f"\nUnseen class sizes:")
unseen_final = df[df["type"] == "unseen"].groupby("class").size().sort_values()
for cls, cnt in unseen_final.items():
    print(f"  {cls:<50} {cnt:>5}")

print(f"\nTotal images : {len(df)}")
print(f"Seen classes : {len(SEEN_CLASSES)}")
print(f"Unseen classes : {len(UNSEEN_CLASSES)}")
print(f"Reserved classes: {len(RESERVED_CLASSES)}")
print("=" * 70)