import os
import glob
import pickle
import datetime
import contextlib
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = r"E:\VS Code\Projects\Thesis\TP002\PlantF\cross_eval_v2"

DEVICE = (torch.device("xpu")
          if (hasattr(torch, "xpu") and torch.xpu.is_available())
          else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

BATCH_SIZE  = 32
IMG_SIZE    = 518
NUM_WORKERS = 0


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg))


class PlantDataset(Dataset):
    def __init__(self, dirs_list, processor):
        self.samples   = []
        self.processor = processor
        for base_dir, split_name in dirs_list:
            if not os.path.exists(base_dir):
                log("  Missing directory: " + base_dir)
                continue
            for cls in sorted(os.listdir(base_dir)):
                cls_path = os.path.join(base_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for img_path in sorted(glob.glob(os.path.join(cls_path, "*.jpg"))):
                    self.samples.append({
                        "path":       img_path,
                        "class_name": cls,
                        "split":      split_name,
                    })
        log("  Dataset size: %d images" % len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        inp = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values": inp["pixel_values"].squeeze(0),
            "class_name":   s["class_name"],
            "path":         s["path"],
            "split":        s["split"],
        }


log("Device: " + str(DEVICE))
log("Loading DINOv2-Large...")


processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-large",
    size={"height": IMG_SIZE, "width": IMG_SIZE},
    do_center_crop=False)

model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(DEVICE)

USE_FP16 = False
try:
    import intel_extension_for_pytorch as ipex
    model    = model.half()
    model    = ipex.optimize(model, dtype=torch.float16)
    USE_FP16 = True
    log("IPEX + FP16 enabled")
except Exception as exc:
    log("IPEX not available: " + str(exc))

model.eval()

if DEVICE.type == "xpu":
    autocast_ctx = torch.xpu.amp.autocast
elif DEVICE.type == "cuda":
    autocast_ctx = torch.cuda.amp.autocast
else:
    autocast_ctx = contextlib.nullcontext


def extract_and_save(direction_dir, direction_label):
    save_dir    = os.path.join(direction_dir, "metadata")
    bundle_path = os.path.join(save_dir, "cross_bundle.pkl")
    os.makedirs(save_dir, exist_ok=True)

    log("Starting extraction: " + direction_label)

    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        if bundle["metadata"] and "type" in bundle["metadata"][0]:
            log("Cache loaded. Shape: " + str(bundle["shape"]))
            return bundle
        else:
            log("Cache invalid — re-extracting")
            os.remove(bundle_path)

    dirs_to_scan = [
        (os.path.join(direction_dir, "source", "train"), "train"),
        (os.path.join(direction_dir, "source", "val"),   "val"),
        (os.path.join(direction_dir, "source", "test"),  "test"),
        (os.path.join(direction_dir, "target", "unseen_test"), "unseen"),
    ]

    dataset = PlantDataset(dirs_to_scan, processor)

    # One-time verification that images are actually reaching the model at
    # IMG_SIZE and not silently center-cropped down. Expect [3, 518, 518].
    if len(dataset) > 0:
        log(f"  Sample pixel_values shape: {tuple(dataset[0]['pixel_values'].shape)}  "
            f"(expect [3, {IMG_SIZE}, {IMG_SIZE}])")

    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True)

    emb_list  = []
    meta_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  " + direction_label):
            pv = batch["pixel_values"].to(DEVICE)
            if USE_FP16:
                pv = pv.half()

            if DEVICE.type in ("xpu", "cuda"):
                with autocast_ctx(enabled=USE_FP16):
                    out  = model(pixel_values=pv)
                    feat = out.last_hidden_state[:, 0, :].float().cpu()
            else:
                out  = model(pixel_values=pv)
                feat = out.last_hidden_state[:, 0, :].float().cpu()

            emb_list.append(feat.numpy())
            for i in range(len(batch["path"])):
                split_val = batch["split"][i]
                img_type  = "unseen" if split_val == "unseen" else "seen"
                meta_list.append({
                    "image_path": batch["path"][i],
                    "class_name": batch["class_name"][i],
                    "split":      split_val,
                    "type":       img_type,
                })
            if DEVICE.type == "xpu":
                torch.xpu.empty_cache()

    embeddings = np.vstack(emb_list)

    n_seen   = sum(1 for m in meta_list if m["type"] == "seen")
    n_unseen = sum(1 for m in meta_list if m["type"] == "unseen")
    log("  Extracted: %d seen  %d unseen  total=%d" % (n_seen, n_unseen, len(meta_list)))

    if n_unseen == 0:
        raise RuntimeError(
            "No unseen images extracted for " + direction_label +
            ". Check target/unseen_test/ exists and contains images.")

    np.save(os.path.join(save_dir, "cross_embeddings.npy"), embeddings)
    pd.DataFrame(meta_list).to_csv(
        os.path.join(save_dir, "cross_metadata.csv"), index=False)

    bundle = {
        "embeddings": embeddings,
        "metadata":   meta_list,
        "model_info": f"dinov2-large-{IMG_SIZE}px",
        "shape":      embeddings.shape,
    }
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    log("Extraction complete. Shape: " + str(embeddings.shape))
    return bundle


dir1_path = os.path.join(OUTPUT_DIR, "direction1_PV_to_PD")
dir2_path = os.path.join(OUTPUT_DIR, "direction2_PD_to_PV")

log("=" * 60)
log("DIRECTION 1: PlantVillage -> PlantDoc")
log("=" * 60)
bundle1 = extract_and_save(dir1_path, "PlantVillage to PlantDoc")

log("=" * 60)
log("DIRECTION 2: PlantDoc -> PlantVillage")
log("=" * 60)
bundle2 = extract_and_save(dir2_path, "PlantDoc to PlantVillage")

log("=" * 60)
log("EXTRACTION COMPLETE")
log("  Direction 1 shape: " + str(bundle1["shape"]))
log("  Direction 2 shape: " + str(bundle2["shape"]))

for label, bundle in [("Direction 1", bundle1), ("Direction 2", bundle2)]:
    types  = [m["type"]  for m in bundle["metadata"]]
    splits = [m["split"] for m in bundle["metadata"]]
    log("  %s: seen=%d  unseen=%d  (train=%d val=%d test=%d unseen=%d)" % (
        label,
        types.count("seen"),
        types.count("unseen"),
        splits.count("train"),
        splits.count("val"),
        splits.count("test"),
        splits.count("unseen")))