import os
import glob
import pickle
import datetime

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

DATA_ROOT  = r"E:\VS Code\Projects\Thesis\TP002\PlantF\Data"
SAVE_DIR   = os.path.join(DATA_ROOT, "metadata")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = (torch.device("xpu")
          if (hasattr(torch, "xpu") and torch.xpu.is_available())
          else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

BATCH_SIZE  = 32
IMG_SIZE    = 518
NUM_WORKERS = 0
LOG_FILE    = os.path.join(SAVE_DIR, "extraction_log.txt")

DIRS_TO_SCAN = [
    (os.path.join(DATA_ROOT, "global", "train"), "train"),
    (os.path.join(DATA_ROOT, "global", "val"),   "val"),
    (os.path.join(DATA_ROOT, "global", "test"),  "test"),
    (os.path.join(DATA_ROOT, "zero_shot", "unseen_test"), "unseen"),
]


def log(msg):
    full = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(full)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full + "\n")


class PlantDataset(Dataset):
    def __init__(self, dirs_list):
        self.samples   = []
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        for base_dir, split_name in dirs_list:
            if not os.path.exists(base_dir):
                log(f"  Missing directory: {base_dir}")
                continue
            for cls in sorted(os.listdir(base_dir)):
                cls_path = os.path.join(base_dir, cls)
                if not os.path.isdir(cls_path):
                    continue
                for img_path in glob.glob(os.path.join(cls_path, "*.jpg")):
                    self.samples.append({
                        "path":       img_path,
                        "class_name": cls,
                        "split":      split_name,
                    })
        log(f"  Total images to process: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        # do_center_crop=False is required here: the default DINOv2 processor
        # resizes to `size` and then center-crops to a separate `crop_size`
        # (224x224 by default) regardless of `size`. Without this, images end
        # up processed at 224x224 no matter what IMG_SIZE is set to.
        inp = self.processor(
            images=img,
            return_tensors="pt",
            size={"height": IMG_SIZE, "width": IMG_SIZE},
            do_center_crop=False)
        return {
            "pixel_values": inp["pixel_values"].squeeze(0),
            "class_name":   s["class_name"],
            "path":         s["path"],
            "split":        s["split"],
        }


log(f"Device: {DEVICE}")
log("Loading DINOv2-Large...")

model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(DEVICE)

USE_FP16 = False
try:
    import intel_extension_for_pytorch as ipex
    model    = model.half()
    model    = ipex.optimize(model, dtype=torch.float16)
    USE_FP16 = True
    log("IPEX + FP16 enabled")
except Exception as e:
    log(f"IPEX not available: {e}")

model.eval()

autocast_ctx = (torch.xpu.amp.autocast  if DEVICE.type == "xpu"  else
                torch.cuda.amp.autocast if DEVICE.type == "cuda" else
                torch.cpu.amp.autocast)

log("Starting feature extraction...")

dataset = PlantDataset(DIRS_TO_SCAN)

# One-time verification: confirms the actual resolution reaching the model.
# Expect torch.Size([3, 518, 518]) here — if you ever see [3, 224, 224]
# again, the processor is still silently center-cropping.
log(f"  Sample pixel_values shape: {tuple(dataset[0]['pixel_values'].shape)}")

loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True)

emb_list  = []
meta_list = []

with torch.no_grad():
    for batch in tqdm(loader, desc="  Extracting"):
        pv = batch["pixel_values"].to(DEVICE)
        if USE_FP16:
            pv = pv.half()
        with autocast_ctx(enabled=USE_FP16):
            out  = model(pixel_values=pv)
            feat = out.last_hidden_state[:, 0, :].float().cpu()
        emb_list.append(feat.numpy())
        for i in range(len(batch["path"])):
            meta_list.append({
                "image_path": batch["path"][i],
                "class_name": batch["class_name"][i],
                "split":      batch["split"][i],
            })
        if DEVICE.type == "xpu":
            torch.xpu.empty_cache()

log("Saving embeddings...")

embeddings = np.vstack(emb_list)
np.save(os.path.join(SAVE_DIR, "PLANT_embeddings.npy"), embeddings)

df = pd.DataFrame(meta_list)
df.to_csv(os.path.join(SAVE_DIR, "PLANT_metadata.csv"), index=False)

bundle = {
    "embeddings": embeddings,
    "metadata":   meta_list,
    "model_info": "dinov2-large-518px",
    "shape":      embeddings.shape,
}
with open(os.path.join(SAVE_DIR, "PLANT_bundle.pkl"), "wb") as f:
    pickle.dump(bundle, f)

log(f"Extraction complete. Embedding shape: {embeddings.shape}")
log(f"Saved to: {SAVE_DIR}")