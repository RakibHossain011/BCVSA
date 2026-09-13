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
OUTPUT_DIR = r"E:\VS Code\Projects\Thesis\TP002\PlantF\cross_eval_v2"

RANDOM_SEED      = 42
TARGET_PER_CLASS = 500
TEST_RATIO       = 0.15
VAL_RATIO        = 0.15
MIN_IMAGES       = 20

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
    "Apple Scab Leaf":            "Apple___Apple_Scab",
    "Apple rust leaf":            "Apple___Cedar_apple_rust",
    "Apple leaf":                 "Apple___Healthy",
    "Blueberry leaf":             "Blueberry___Healthy",
    "Cherry leaf":                "Cherry___Healthy",
    "Cherry Powdery Mildew":      "Cherry___Powdery_mildew",
    "Corn Gray leaf spot":        "Corn___Gray_leaf_spot",
    "Corn leaf blight":           "Corn___Northern_Leaf_Blight",
    "Corn rust leaf":             "Corn___Common_rust",
    "grape leaf":                 "Grape___Healthy",
    "grape leaf black rot":       "Grape___Black_rot",
    "Peach leaf":                 "Peach___Healthy",
    "Potato leaf early blight":   "Potato___Early_blight",
    "Potato leaf late blight":    "Potato___Late_blight",
    "Raspberry leaf":             "Raspberry___Healthy",
    "Soyabean leaf":              "Soybean___Healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf":            "Strawberry___Healthy",
    "Tomato Early blight leaf":   "Tomato___Early_blight",
    "Tomato leaf":                "Tomato___Healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight":    "Tomato___Late_blight",
    "Tomato leaf mosaic virus":   "Tomato___Mosaic_Virus",
    "Tomato leaf yellow virus":   "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf":           "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot":  "Tomato___Septoria_leaf_spot",
    "Bell_pepper leaf":           "Pepper_Bell___Healthy",
    "Bell_pepper leaf spot":      "Pepper_Bell___Bacterial_spot",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_valid(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def md5_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_dir(p):
    os.makedirs(p, exist_ok=True)


def copy_img(src, dst_dir, fname):
    name = os.path.splitext(fname)[0] + ".jpg"
    dst  = os.path.join(dst_dir, name)
    try:
        img = Image.open(src).convert("RGB")
        img.save(dst, "JPEG", quality=95)
        return name
    except Exception as exc:
        print("  WARNING copy failed %s: %s" % (src, exc))
        return None


print("\n[Step 1] Collecting images...")

pv_raw = defaultdict(list)
pd_raw = defaultdict(list)

pv_path = os.path.join(RAW_DIR, "plantvillage")
pd_path = os.path.join(RAW_DIR, "plantdoc")

if not os.path.exists(pv_path):
    raise FileNotFoundError("PlantVillage not found: " + pv_path)
if not os.path.exists(pd_path):
    raise FileNotFoundError("PlantDoc not found: " + pd_path)

for root, dirs, files in os.walk(pv_path):
    folder  = os.path.basename(root)
    unified = RAW_TO_UNIFIED.get(folder)
    if unified is None:
        continue
    imgs = [os.path.join(root, f) for f in files
            if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if imgs:
        pv_raw[unified].extend(imgs)

for root, dirs, files in os.walk(pd_path):
    folder  = os.path.basename(root)
    unified = RAW_TO_UNIFIED.get(folder)
    if unified is None:
        continue
    imgs = [os.path.join(root, f) for f in files
            if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if imgs:
        pd_raw[unified].extend(imgs)

print("  PlantVillage raw classes: %d" % len(pv_raw))
print("  PlantDoc    raw classes : %d" % len(pd_raw))


print("\n[Step 2] Deduplication and corrupt removal...")


def clean_dataset(raw_dict, label):
    hash_reg  = {}
    removed_c = 0
    removed_d = 0
    clean     = defaultdict(list)
    for cls in tqdm(sorted(raw_dict), desc="  Cleaning " + label):
        for path in raw_dict[cls]:
            if not is_valid(path):
                removed_c += 1
                continue
            h = md5_hash(path)
            if h in hash_reg:
                removed_d += 1
                continue
            hash_reg[h] = path
            clean[cls].append(path)
    print("  [%s] corrupt=%d  duplicates=%d" % (label, removed_c, removed_d))
    return clean


pv_clean = clean_dataset(pv_raw, "PlantVillage")
pd_clean = clean_dataset(pd_raw, "PlantDoc")

pd_filtered = {cls: paths for cls, paths in pd_clean.items()
               if len(paths) >= MIN_IMAGES}

print("\n  PlantVillage clean classes: %d" % len(pv_clean))
for cls in sorted(pv_clean):
    print("    %-58s %5d" % (cls, len(pv_clean[cls])))

print("\n  PlantDoc clean classes (minimum %d images): %d" % (MIN_IMAGES, len(pd_filtered)))
for cls in sorted(pd_filtered):
    print("    %-58s %5d" % (cls, len(pd_filtered[cls])))


print("\n[Step 3] Finding shared classes...")

pv_cls_set     = set(pv_clean.keys())
pd_cls_set     = set(pd_filtered.keys())
shared_classes = sorted(pv_cls_set & pd_cls_set)

if len(shared_classes) == 0:
    raise RuntimeError("No shared classes found. Check RAW_TO_UNIFIED and folder names.")

print("  PlantVillage classes : %d" % len(pv_cls_set))
print("  PlantDoc    classes  : %d" % len(pd_cls_set))
print("  Shared classes       : %d" % len(shared_classes))

print("\n  Shared classes:")
for cls in shared_classes:
    print("    %-58s  PV=%5d  PD=%5d" % (
        cls, len(pv_clean[cls]), len(pd_filtered[cls])))


print("\n[Step 4] Balancing and splitting...")


def balance_and_split(class_dict, label):
    splits = {}
    report = []
    for cls in sorted(class_dict.keys()):
        images = class_dict[cls]
        n_raw  = len(images)
        if n_raw < MIN_IMAGES:
            print("  [%s] SKIP %s: %d images" % (label, cls, n_raw))
            report.append({"class": cls, "raw": n_raw, "kept": 0, "note": "skipped"})
            continue
        if n_raw > TARGET_PER_CLASS:
            kept = random.sample(images, TARGET_PER_CLASS)
            note = "downsampled %d -> %d" % (n_raw, TARGET_PER_CLASS)
        else:
            kept = list(images)
            note = "all %d kept" % n_raw
        tr_val, te = train_test_split(
            kept, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        tr, va = train_test_split(
            tr_val,
            test_size=VAL_RATIO / (1.0 - TEST_RATIO),
            random_state=RANDOM_SEED)
        splits[cls] = {"train": tr, "val": va, "test": te}
        report.append({"class": cls, "raw": n_raw, "kept": len(kept), "note": note})
        print("  [%s] %-52s %5d -> %5d  (%s)" % (label, cls, n_raw, len(kept), note))
    tr_tot = sum(len(v["train"]) for v in splits.values())
    va_tot = sum(len(v["val"])   for v in splits.values())
    te_tot = sum(len(v["test"])  for v in splits.values())
    print("  [%s] train=%d  val=%d  test=%d" % (label, tr_tot, va_tot, te_tot))
    return splits, report


pv_splits, pv_report = balance_and_split(pv_clean,   "PlantVillage")
pd_splits, pd_report = balance_and_split(pd_filtered, "PlantDoc")


print("\n[Step 5] Building output folders...")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

dir1 = os.path.join(OUTPUT_DIR, "direction1_PV_to_PD")
dir2 = os.path.join(OUTPUT_DIR, "direction2_PD_to_PV")

for d in [dir1, dir2]:
    for split in ["train", "val", "test"]:
        make_dir(os.path.join(d, "source", split))

    make_dir(os.path.join(d, "target", "unseen_test"))
    make_dir(os.path.join(d, "metadata"))


print("\n[Step 6a] Direction 1: PlantVillage source -> PlantDoc target...")

meta1       = []
seen1_cls   = sorted(pv_splits.keys())
unseen1_cls = shared_classes
all1_cls    = sorted(set(seen1_cls) | set(unseen1_cls))
cls_to_id1  = {c: i for i, c in enumerate(all1_cls)}

for cls in tqdm(sorted(pv_splits.keys()), desc="  PlantVillage source"):
    for split_name, img_list in pv_splits[cls].items():
        dst = os.path.join(dir1, "source", split_name, cls)
        make_dir(dst)
        for i, img in enumerate(img_list):
            fname = copy_img(img, dst, cls + "_" + split_name + "_%05d.jpg" % i)
            if fname:
                meta1.append({
                    "image_path": os.path.join(dst, fname),
                    "class_name": cls,
                    "class_id":   cls_to_id1.get(cls, -1),
                    "crop":       cls.split("___")[0],
                    "type":       "seen",
                    "split":      split_name,
                    "dataset":    "PlantVillage",
                })

for cls in tqdm(shared_classes, desc="  PlantDoc target"):
    dst = os.path.join(dir1, "target", "unseen_test", cls)
    make_dir(dst)
    for i, img in enumerate(pd_filtered[cls]):
        fname = copy_img(img, dst, cls + "_%05d.jpg" % i)
        if fname:
            meta1.append({
                "image_path": os.path.join(dst, fname),
                "class_name": cls,
                "class_id":   cls_to_id1.get(cls, -1),
                "crop":       cls.split("___")[0],
                "type":       "unseen",
                "split":      "unseen_test",
                "dataset":    "PlantDoc",
            })

df1 = pd.DataFrame(meta1)
df1.to_csv(os.path.join(dir1, "metadata", "complete_metadata.csv"), index=False)
with open(os.path.join(dir1, "metadata", "class_mapping.json"), "w") as f:
    json.dump(cls_to_id1, f, indent=2)
with open(os.path.join(dir1, "metadata", "shared_classes.json"), "w") as f:
    json.dump({
        "shared_classes": shared_classes,
        "seen_classes":   seen1_cls,
        "unseen_classes": unseen1_cls,
    }, f, indent=2)

d1s = df1[df1["type"] == "seen"]
d1u = df1[df1["type"] == "unseen"]
print("  Direction 1 source (PlantVillage) : %d classes  %d images" % (d1s["class_name"].nunique(), len(d1s)))
print("  Direction 1 target (PlantDoc)     : %d classes  %d images" % (d1u["class_name"].nunique(), len(d1u)))


print("\n[Step 6b] Direction 2: PlantDoc source -> PlantVillage target...")

meta2       = []
seen2_cls   = sorted(pd_splits.keys())
unseen2_cls = shared_classes
all2_cls    = sorted(set(seen2_cls) | set(unseen2_cls))
cls_to_id2  = {c: i for i, c in enumerate(all2_cls)}

for cls in tqdm(sorted(pd_splits.keys()), desc="  PlantDoc source"):
    for split_name, img_list in pd_splits[cls].items():
        dst = os.path.join(dir2, "source", split_name, cls)
        make_dir(dst)
        for i, img in enumerate(img_list):
            fname = copy_img(img, dst, cls + "_" + split_name + "_%05d.jpg" % i)
            if fname:
                meta2.append({
                    "image_path": os.path.join(dst, fname),
                    "class_name": cls,
                    "class_id":   cls_to_id2.get(cls, -1),
                    "crop":       cls.split("___")[0],
                    "type":       "seen",
                    "split":      split_name,
                    "dataset":    "PlantDoc",
                })

for cls in tqdm(shared_classes, desc="  PlantVillage target"):
    dst = os.path.join(dir2, "target", "unseen_test", cls)
    make_dir(dst)
    n_available = len(pv_clean[cls])
    img_list_capped = random.sample(pv_clean[cls], min(TARGET_PER_CLASS, n_available))
    for i, img in enumerate(img_list_capped):
        fname = copy_img(img, dst, cls + "_%05d.jpg" % i)
        if fname:
            meta2.append({
                "image_path": os.path.join(dst, fname),
                "class_name": cls,
                "class_id":   cls_to_id2.get(cls, -1),
                "crop":       cls.split("___")[0],
                "type":       "unseen",
                "split":      "unseen_test",
                "dataset":    "PlantVillage",
            })

df2 = pd.DataFrame(meta2)
df2.to_csv(os.path.join(dir2, "metadata", "complete_metadata.csv"), index=False)
with open(os.path.join(dir2, "metadata", "class_mapping.json"), "w") as f:
    json.dump(cls_to_id2, f, indent=2)
with open(os.path.join(dir2, "metadata", "shared_classes.json"), "w") as f:
    json.dump({
        "shared_classes": shared_classes,
        "seen_classes":   seen2_cls,
        "unseen_classes": unseen2_cls,
    }, f, indent=2)

d2s = df2[df2["type"] == "seen"]
d2u = df2[df2["type"] == "unseen"]
print("  Direction 2 source (PlantDoc)      : %d classes  %d images" % (d2s["class_name"].nunique(), len(d2s)))
print("  Direction 2 target (PlantVillage)  : %d classes  %d images" % (d2u["class_name"].nunique(), len(d2u)))


pd.DataFrame(pv_report).to_csv(
    os.path.join(OUTPUT_DIR, "plantvillage_balance_report.csv"), index=False)
pd.DataFrame(pd_report).to_csv(
    os.path.join(OUTPUT_DIR, "plantdoc_balance_report.csv"), index=False)

print("\n" + "=" * 70)
print("CROSS-DATASET DATA PREPARATION COMPLETE")
print("=" * 70)
print("Output: " + OUTPUT_DIR)
print("\nShared classes (%d) -- note: for Direction 2 the source class set is" % len(shared_classes))
print("*identical* to this shared set (PlantDoc's own mapping only covers")
print("these categories), so Direction 2 trains and evaluates on the exact")
print("same taxonomy -- this is cross-DOMAIN evaluation, not cross-CLASS.")
for cls in shared_classes:
    print("  %-58s  PV=%5d  PD=%5d" % (
        cls, len(pv_clean[cls]), len(pd_filtered[cls])))
print("\nDirection 1 — PlantVillage source -> PlantDoc target")
print("  Source: %d classes  %d images" % (d1s["class_name"].nunique(), len(d1s)))
print("  Target: %d classes  %d images" % (d1u["class_name"].nunique(), len(d1u)))
print("\nDirection 2 — PlantDoc source -> PlantVillage target")
print("  Source: %d classes  %d images" % (d2s["class_name"].nunique(), len(d2s)))
print("  Target: %d classes  %d images" % (d2u["class_name"].nunique(), len(d2u)))
print("=" * 70)