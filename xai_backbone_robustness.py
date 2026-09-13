import os
import glob
import pickle
import math
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import label as scipy_label
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

DATA_ROOT       = r"E:\VS Code\Projects\Thesis\TP002\PlantF\Data"
SAVE_DIR        = os.path.join(DATA_ROOT, "metadata")
RESULTS_DIR     = os.path.join(DATA_ROOT, "metadata", "results")
TRAIN_DIR       = os.path.join(DATA_ROOT, "global", "train")
SEEN_TEST_DIR   = os.path.join(DATA_ROOT, "global", "test")
UNSEEN_TEST_DIR = os.path.join(DATA_ROOT, "zero_shot", "unseen_test")
BUNDLE_PATH     = os.path.join(SAVE_DIR, "PLANT_bundle.pkl")
PROBE_BEST      = os.path.join(SAVE_DIR, "probe_best.pt")
os.makedirs(RESULTS_DIR, exist_ok=True)

GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

DEVICE = (torch.device("xpu")  if hasattr(torch, "xpu") and torch.xpu.is_available() else
          torch.device("cuda")  if torch.cuda.is_available() else
          torch.device("cpu"))
print(f"Device: {DEVICE}")

t0 = time.time()
def elapsed():
    return f"{(time.time()-t0)/60:.1f} min"

def section(msg):
    print(f"\n{'='*70}\n{msg}  ({elapsed()})\n{'='*70}")

section("Loading artefacts")

with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

meta = pd.DataFrame(bundle["metadata"])
meta.index      = range(len(meta))
meta["pos_idx"] = meta.index
if "type" not in meta.columns:
    meta["type"] = "seen"
    meta.loc[meta["split"].str.contains("unseen", case=False, na=False), "type"] = "unseen"

dino_emb_large = F.normalize(
    torch.tensor(bundle["embeddings"], dtype=torch.float32), dim=-1)
print(f"  dino_emb_large: {dino_emb_large.shape}")

seen_classes   = sorted([d for d in os.listdir(TRAIN_DIR)
                          if os.path.isdir(os.path.join(TRAIN_DIR, d))])
unseen_classes = sorted([d for d in os.listdir(UNSEEN_TEST_DIR)
                          if os.path.isdir(os.path.join(UNSEEN_TEST_DIR, d))])
all_classes    = sorted(list(set(seen_classes) | set(unseen_classes)))
cls2id         = {c: i for i, c in enumerate(all_classes)}
seen_ids_np    = np.array([cls2id[c] for c in seen_classes])
unseen_ids_np  = np.array([cls2id[c] for c in unseen_classes])
seen_local     = {c: i for i, c in enumerate(seen_classes)}
all_cls_arr    = np.array(all_classes)
unseen_arr     = np.array(unseen_classes)
S_n = len(seen_classes)
C   = len(all_classes)
print(f"  Seen={S_n}  Unseen={len(unseen_classes)}  All={C}")

stest_meta = meta[(meta["type"] == "seen") & (meta["split"] == "test")].copy()
val_meta   = meta[(meta["type"] == "seen") & (meta["split"] == "val")].copy()
train_meta = meta[(meta["type"] == "seen") & (meta["split"] == "train")].copy()
utest_meta = meta[meta["type"] == "unseen"].copy()
train_meta["seen_id"] = train_meta["class_name"].map(seen_local)
val_meta["seen_id"]   = val_meta["class_name"].map(seen_local)

s_labels_np  = np.array(stest_meta["class_name"].tolist())
u_lab_all_np = np.array(utest_meta["class_name"].tolist())

TEXT_CACHE   = os.path.join(SAVE_DIR, "clip_text_prototypes.pt")
all_text_cpu = torch.load(TEXT_CACHE, weights_only=False)
all_text_np  = all_text_cpu.numpy()
print(f"  Text prototypes: {all_text_cpu.shape}")

CLIP_U_CACHE     = os.path.join(SAVE_DIR, "clip_features_unseen.pt")
clip_cf_to_idx   = {}
clip_unseen_feat = None
if os.path.exists(CLIP_U_CACHE):
    d = torch.load(CLIP_U_CACHE, weights_only=False)
    clip_unseen_feat = d["feat"]
    clip_cf = d.get("class_filenames", None)
    if clip_cf:
        clip_cf_to_idx = {cf: i for i, cf in enumerate(clip_cf)}
    print(f"  CLIP unseen cache: {clip_unseen_feat.shape}")


class LinearProbe(nn.Module):
    def __init__(self, in_dim=1024, out=S_n):
        super().__init__()
        mid1 = max(64, in_dim // 2)
        mid2 = max(32, mid1 // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid1), nn.LayerNorm(mid1), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mid1,  mid2), nn.LayerNorm(mid2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(mid2,  out))
    def forward(self, x):
        return self.net(x)


probe = LinearProbe(in_dim=1024, out=S_n).to(DEVICE)
probe.load_state_dict(torch.load(PROBE_BEST, map_location=DEVICE, weights_only=False))
probe.eval()
print("  Probe loaded.")


def softmax_np(x, T=1.0):
    x = np.asarray(x, dtype=np.float64) / T
    e = np.exp(x - x.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)

def hm(s, u):
    return 2 * s * u / (s + u + 1e-9)

def pacc(p, y):
    return float((np.asarray(p) == np.asarray(y)).mean())

def expand_full(sm, N, C, seen_ids):
    full = np.zeros((N, C), dtype=np.float32)
    for j, si in enumerate(seen_ids):
        full[:, si] = sm[:, j]
    return full

def get_probe_sm(emb_tensor, idx_arr, model, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_arr), batch):
            idx = idx_arr[i:i+batch]
            v   = emb_tensor[idx].to(DEVICE)
            out.append(F.softmax(model(v), dim=-1).cpu().numpy())
    return np.vstack(out)

def gzsl_eval(fs, fu, y_s, y_u, cls_arr):
    pred_s = cls_arr[fs.argmax(1)]
    pred_u = cls_arr[fu.argmax(1)]
    S = pacc(pred_s, y_s)
    U = pacc(pred_u, y_u)
    return pred_s, pred_u, S, U, hm(S, U)

def savefig(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return path

def shorten(s):
    return s.replace("___", " — ").replace("_", " ")

def _trunc(s, limit=28):

    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0]
    return (cut if cut else s[:limit]) + "…"


section("[A] XAI — DINOv2 Attention Maps")

IMG_SIZE  = 518
ALPHA_OVL = 0.50

print("  Loading DINOv2-Large...")

dino_proc  = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-large",
    size={"height": IMG_SIZE, "width": IMG_SIZE},
    do_center_crop=False)
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(DEVICE)
dino_model.eval()
try:
    import intel_extension_for_pytorch as ipex
    dino_model = ipex.optimize(dino_model)
    print("  IPEX enabled")
except Exception:
    pass


def remove_register_artifacts(attn_grid):
    grid = attn_grid.copy().astype(np.float64)
    H, W = grid.shape
    mu, sigma  = grid.mean(), grid.std()
    threshold  = mu + 3.0 * sigma
    spike_mask = grid > threshold
    labeled, n_components = scipy_label(spike_mask)
    n_artifacts = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if int(comp_mask.sum()) > 6:
            continue
        r = 4
        rows, cols = np.where(comp_mask)
        r_min = max(0, rows.min() - r);  r_max = min(H, rows.max() + r + 1)
        c_min = max(0, cols.min() - r);  c_max = min(W, cols.max() + r + 1)
        region     = grid[r_min:r_max, c_min:c_max].copy()
        local_mask = comp_mask[r_min:r_max, c_min:c_max]
        surrounding = region[~local_mask]
        grid[comp_mask] = float(np.median(surrounding)) if len(surrounding) else mu
        n_artifacts += 1
    p_lo, p_hi = np.percentile(grid, 5), np.percentile(grid, 95)
    cleaned = np.clip((grid - p_lo) / (p_hi - p_lo + 1e-8), 0., 1.)
    return cleaned.astype(np.float32), n_artifacts


def clip_top2_unseen(img_path, cls_name):
    cf_key = f"{cls_name}/{os.path.basename(img_path)}"
    if clip_unseen_feat is not None and cf_key in clip_cf_to_idx:
        idx  = clip_cf_to_idx[cf_key]
        feat = clip_unseen_feat[idx:idx+1]
        sims = (feat @ all_text_cpu.T).numpy()[0]
        top2 = sims.argsort()[::-1][:2]
        return [(all_classes[i], float(sims[i])) for i in top2]
    return [("(not in cache)", 0.), ("(not in cache)", 0.)]


def get_clean_attention(img_path, cls_name, is_unseen):
    orig   = Image.open(img_path).convert("RGB")
    inputs = dino_proc(images=orig, return_tensors="pt")
    pv     = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        out = dino_model(pixel_values=pv, output_attentions=True)
    last_attn   = out.attentions[-1]
    seq_len     = last_attn.shape[-1]
    num_patches = seq_len - 1
    n_side      = int(math.isqrt(num_patches))
    assert n_side * n_side == num_patches
    cls_attn  = last_attn[0, :, 0, 1:].cpu().float()
    mean_attn = cls_attn.mean(0).numpy()
    attn_grid = mean_attn.reshape(n_side, n_side)
    attn_clean_grid, n_art = remove_register_artifacts(attn_grid)
    print(f"    {n_side}×{n_side} patches  |  {n_art} artifact spike(s) inpainted")
    attn_t  = torch.tensor(attn_clean_grid).unsqueeze(0).unsqueeze(0)
    attn_up = F.interpolate(attn_t, size=(IMG_SIZE, IMG_SIZE),
                             mode="bicubic", align_corners=False).squeeze().numpy()
    attn_clean = np.clip(attn_up, 0., 1.).astype(np.float32)
    cls_emb = F.normalize(out.last_hidden_state[:, 0, :].float().cpu(), dim=-1)
    with torch.no_grad():
        logits = probe(cls_emb.to(DEVICE)).cpu()
    scores = F.softmax(logits, dim=-1).squeeze(0).numpy()
    top2_i     = scores.argsort()[::-1][:2]
    top2_probe = [(seen_classes[i], float(scores[i])) for i in top2_i]
    top2_clip  = clip_top2_unseen(img_path, cls_name) if is_unseen else []
    orig_rgb   = np.array(orig.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
    return orig_rgb, attn_clean, top2_probe, top2_clip, n_art


def pick_best_image(cls_path, cls_name, is_unseen, n_candidates=15):
    imgs = sorted(glob.glob(os.path.join(cls_path, "*.jpg")))[:n_candidates]
    if len(imgs) <= 1:
        return imgs[0] if imgs else None
    best_img, best_score = imgs[0], -1.
    for img_path in imgs:
        try:
            orig = Image.open(img_path).convert("RGB")
            inp  = dino_proc(images=orig, return_tensors="pt")
            with torch.no_grad():
                out  = dino_model(pixel_values=inp["pixel_values"].to(DEVICE))
                emb  = F.normalize(out.last_hidden_state[:, 0, :].float(), dim=-1)
                if is_unseen:
                    cf_key = f"{cls_name}/{os.path.basename(img_path)}"
                    if clip_unseen_feat is not None and cf_key in clip_cf_to_idx:
                        idx   = clip_cf_to_idx[cf_key]
                        feat  = clip_unseen_feat[idx:idx+1]
                        ci    = cls2id[cls_name]
                        score = float((feat @ all_text_cpu[ci:ci+1].T).item())
                    else:
                        score = 0.
                else:
                    probs = F.softmax(probe(emb), dim=-1).squeeze(0)
                    si    = seen_local.get(cls_name, -1)
                    score = float(probs[si]) if si >= 0 else 0.
            if score > best_score:
                best_score, best_img = score, img_path
        except Exception:
            continue
    return best_img


XAI_TARGETS = [
    (SEEN_TEST_DIR,   "Apple___Apple_Scab",         "Seen — Disease",   False),
    (SEEN_TEST_DIR,   "Corn___Common_rust",          "Seen — Disease",   False),
    (SEEN_TEST_DIR,   "Tomato___Septoria_leaf_spot", "Seen — Disease",   False),
    (SEEN_TEST_DIR,   "Tomato___Healthy",            "Seen — Healthy",   False),
    (UNSEEN_TEST_DIR, "Orange___Citrus_greening",    "Unseen — Disease", True),
    (UNSEEN_TEST_DIR, "Squash___Powdery_mildew",     "Unseen — Disease", True),
]

_sample_path = None
for base_dir, cls_name, _lbl, _u in XAI_TARGETS:
    _p = os.path.join(base_dir, cls_name)
    if os.path.isdir(_p):
        _imgs = glob.glob(os.path.join(_p, "*.jpg"))
        if _imgs:
            _sample_path = _imgs[0]
            break
if _sample_path:
    _sample_inp = dino_proc(images=Image.open(_sample_path).convert("RGB"), return_tensors="pt")
    print(f"  Sample pixel_values shape: {tuple(_sample_inp['pixel_values'].shape)}  "
          f"(expect [1, 3, {IMG_SIZE}, {IMG_SIZE}])")

print("\n  Processing images...")
xai_results = []
for base_dir, cls_name, split_label, is_unseen in XAI_TARGETS:
    cls_path = os.path.join(base_dir, cls_name)
    if not os.path.isdir(cls_path):
        print(f"  SKIP: {cls_path} not found")
        continue
    print(f"  {cls_name}")
    chosen = pick_best_image(cls_path, cls_name, is_unseen)
    if chosen is None:
        print(f"  SKIP: no images")
        continue
    orig_rgb, attn_clean, top2_probe, top2_clip, n_art = \
        get_clean_attention(chosen, cls_name, is_unseen)
    xai_results.append(dict(cls_name=cls_name, split_label=split_label,
                            is_unseen=is_unseen, orig_rgb=orig_rgb,
                            attn=attn_clean, top2_probe=top2_probe,
                            top2_clip=top2_clip, n_art=n_art))

del dino_model
if DEVICE.type == "xpu":    torch.xpu.empty_cache()
elif DEVICE.type == "cuda": torch.cuda.empty_cache()

cmap_attn  = cm.get_cmap("inferno")
n_img      = len(xai_results)
seen_count = sum(1 for r in xai_results if not r["is_unseen"])

fig, axes = plt.subplots(3, n_img, figsize=(4.3 * n_img, 11.5), squeeze=False)

last_attn_im = None
for col, res in enumerate(xai_results):
    orig      = res["orig_rgb"]
    attn      = res["attn"]
    is_unseen = res["is_unseen"]
    top2_p    = res["top2_probe"]
    top2_c    = res["top2_clip"]
    border_col = "#DC2626" if is_unseen else "#2563EB"

    # Row 0: original image
    ax0 = axes[0, col]
    ax0.imshow(orig)
    ax0.set_title(shorten(res["cls_name"]), fontsize=9, fontweight="bold", pad=6)
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.spines["top"].set_visible(True)
    ax0.spines["top"].set_linewidth(3.5)
    ax0.spines["top"].set_color(border_col)
    if col == 0:
        ax0.set_ylabel("Original", fontsize=9, fontweight="bold")

    # Row 1: attention map
    ax1 = axes[1, col]
    last_attn_im = ax1.imshow(attn, cmap="inferno", vmin=0, vmax=1)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title(f"{res['n_art']} spike(s) inpainted", fontsize=7, color="#6B7280")
    if col == 0:
        ax1.set_ylabel("CLS Attention", fontsize=9, fontweight="bold")

    # Row 2: overlay + prediction
    ax2 = axes[2, col]
    attn_rgba = cmap_attn(attn)
    attn_rgb  = (attn_rgba[:, :, :3] * 255).astype(np.uint8)
    overlay   = np.clip(ALPHA_OVL * attn_rgb + (1 - ALPHA_OVL) * orig, 0, 255).astype(np.uint8)
    ax2.imshow(overlay)
    ax2.set_xticks([]); ax2.set_yticks([])
    if col == 0:
        ax2.set_ylabel("Overlay", fontsize=9, fontweight="bold")

    if is_unseen:
        pred_lines = (
            f"CLIP (all {C} classes):\n"
            f"① {_trunc(shorten(top2_c[0][0]))}  {top2_c[0][1]*100:.1f}%\n"
            f"② {_trunc(shorten(top2_c[1][0]))}  {top2_c[1][1]*100:.1f}%")
        txt_col = "#DC2626"
    else:
        pred_lines = (
            f"Probe (seen {S_n} classes):\n"
            f"① {_trunc(shorten(top2_p[0][0]))}  {top2_p[0][1]*100:.1f}%\n"
            f"② {_trunc(shorten(top2_p[1][0]))}  {top2_p[1][1]*100:.1f}%")
        txt_col = "#16A34A"
    ax2.set_xlabel(pred_lines, fontsize=6.8, color=txt_col, loc="left")

# Vertical divider between the seen-class columns and unseen-class columns.
if 0 < seen_count < n_img:
    for r in range(3):
        ax = axes[r, seen_count - 1]
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_linewidth(2.5)
        ax.spines["right"].set_color("#9CA3AF")
        ax.spines["right"].set_linestyle("--")

if 0 < seen_count < n_img:
    seen_x   = (seen_count / 2) / n_img
    unseen_x = (seen_count + (n_img - seen_count) / 2) / n_img
    fig.text(seen_x,   0.895, "SEEN CLASSES",   ha="center", fontsize=9,
              color="#2563EB", fontweight="bold")
    fig.text(unseen_x, 0.895, "UNSEEN CLASSES", ha="center", fontsize=9,
              color="#DC2626", fontweight="bold")


fig.suptitle(
    "DINOv2-Large CLS Attention Maps — What the Model Focuses On\n"
    "Register-token spikes spatially inpainted. "
    "Seen classes: probe softmax.  Unseen classes: CLIP cosine similarity.",
    fontsize=10, fontweight="bold", y=0.975)


fig.subplots_adjust(left=0.035, right=0.90, top=0.84, bottom=0.10,
                     wspace=0.12, hspace=0.18)

if last_attn_im is not None:
    cax = fig.add_axes([0.915, 0.40, 0.012, 0.22])  # [left, bottom, width, height]
    cbar = fig.colorbar(last_attn_im, cax=cax)
    cbar.set_label("Relative attention", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

savefig(fig, "figure10_xai_attention_final.png")


section("[B] Backbone Robustness — DINOv2-Small vs DINOv2-Large")

SMALL_EMB_PATH   = os.path.join(SAVE_DIR, "PLANT_embeddings_small.npy")
SMALL_META_PATH  = os.path.join(SAVE_DIR, "PLANT_metadata_small.csv")
SMALL_PROBE_BEST = os.path.join(SAVE_DIR, "probe_small_best.pt")
SMALL_DIM        = 384
BATCH_SIZE       = 32
IMG_SIZE_SMALL   = 518

if os.path.exists(SMALL_EMB_PATH):
    print(f"  Loading cached small embeddings...")
    print(f"  NOTE: if this cache predates the do_center_crop fix below, it")
    print(f"  was extracted at 224px, not {IMG_SIZE_SMALL}px -- delete "
          f"{os.path.basename(SMALL_EMB_PATH)}, "
          f"{os.path.basename(SMALL_META_PATH)}, and "
          f"{os.path.basename(SMALL_PROBE_BEST)} and rerun if unsure.")
    emb_small_np = np.load(SMALL_EMB_PATH)
    small_meta   = pd.read_csv(SMALL_META_PATH)
    print(f"  Shape: {emb_small_np.shape}")
else:
    print("  Extracting DINOv2-Small features...")
    # Same do_center_crop=False fix as the large backbone above, for a fair
    # apples-to-apples comparison at the same input resolution.
    proc_small  = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-small",
        size={"height": IMG_SIZE_SMALL, "width": IMG_SIZE_SMALL},
        do_center_crop=False)
    model_small = Dinov2Model.from_pretrained("facebook/dinov2-small").to(DEVICE)
    USE_FP16_S  = False
    try:
        import intel_extension_for_pytorch as ipex
        model_small = model_small.half()
        model_small = ipex.optimize(model_small, dtype=torch.float16)
        USE_FP16_S  = True
        print("  IPEX FP16 enabled for small model")
    except Exception:
        pass
    model_small.eval()
    dirs_to_scan = [
        (os.path.join(DATA_ROOT, "global", "train"),          "train"),
        (os.path.join(DATA_ROOT, "global", "val"),            "val"),
        (os.path.join(DATA_ROOT, "global", "test"),           "test"),
        (os.path.join(DATA_ROOT, "zero_shot", "unseen_test"), "unseen"),
    ]
    image_list = []
    for base_dir, split_name in dirs_to_scan:
        if not os.path.exists(base_dir):
            continue
        for cls in sorted(os.listdir(base_dir)):
            cls_path = os.path.join(base_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_path in sorted(glob.glob(os.path.join(cls_path, "*.jpg"))):
                image_list.append({"path": img_path, "class_name": cls, "split": split_name})
    print(f"  Total images: {len(image_list)}")

    # One-time verification, same as the large-backbone check above.
    if image_list:
        _sample_inp = proc_small(images=Image.open(image_list[0]["path"]).convert("RGB"),
                                  return_tensors="pt")
        print(f"  Sample pixel_values shape: {tuple(_sample_inp['pixel_values'].shape)}  "
              f"(expect [1, 3, {IMG_SIZE_SMALL}, {IMG_SIZE_SMALL}])")

    autocast_ctx = (torch.xpu.amp.autocast  if DEVICE.type == "xpu"  else
                    torch.cuda.amp.autocast  if DEVICE.type == "cuda" else
                    torch.cpu.amp.autocast)
    emb_list_s, meta_list_s = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_list), BATCH_SIZE), desc="  Extracting"):
            batch_items = image_list[i:i+BATCH_SIZE]
            imgs = []
            for item in batch_items:
                try:
                    imgs.append(Image.open(item["path"]).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (IMG_SIZE_SMALL, IMG_SIZE_SMALL), (128, 128, 128)))
            inp  = proc_small(images=imgs, return_tensors="pt")
            pv   = inp["pixel_values"].to(DEVICE)
            if USE_FP16_S:
                pv = pv.half()
            with autocast_ctx(enabled=USE_FP16_S):
                out  = model_small(pixel_values=pv)
                feat = out.last_hidden_state[:, 0, :].float().cpu()
            emb_list_s.append(feat.numpy())
            meta_list_s.extend(batch_items)
            if DEVICE.type == "xpu":
                torch.xpu.empty_cache()
    emb_small_np = np.vstack(emb_list_s)
    np.save(SMALL_EMB_PATH, emb_small_np)
    small_meta = pd.DataFrame(meta_list_s)
    small_meta.to_csv(SMALL_META_PATH, index=False)
    print(f"  Saved. Shape: {emb_small_np.shape}")
    del model_small

dino_emb_small = F.normalize(
    torch.tensor(emb_small_np, dtype=torch.float32), dim=-1)
print(f"  dino_emb_small: {dino_emb_small.shape}")

small_meta["pos_idx"] = range(len(small_meta))
s_tr   = small_meta[small_meta["split"] == "train"].copy()
s_val  = small_meta[small_meta["split"] == "val"].copy()
s_test = small_meta[small_meta["split"] == "test"].copy()
s_uns  = small_meta[small_meta["split"] == "unseen"].copy()
s_tr["seen_id"]  = s_tr["class_name"].map(seen_local)
s_val["seen_id"] = s_val["class_name"].map(seen_local)
s_labels_small   = np.array(s_test["class_name"].tolist())
v_labels_small   = np.array(s_val["class_name"].tolist())
print(f"  Small splits: train={len(s_tr)}  val={len(s_val)}  "
      f"test={len(s_test)}  unseen={len(s_uns)}")

EPOCHS_S = 200
LR_S     = 3e-4
BATCH_S  = 512

if os.path.exists(SMALL_PROBE_BEST):
    print("  Loading cached small probe...")
    probe_small = LinearProbe(in_dim=SMALL_DIM, out=S_n).to(DEVICE)
    probe_small.load_state_dict(
        torch.load(SMALL_PROBE_BEST, map_location=DEVICE, weights_only=False))
    probe_small.eval()
else:
    print(f"  Training small probe ({EPOCHS_S} epochs)...")
    probe_small = LinearProbe(in_dim=SMALL_DIM, out=S_n).to(DEVICE)
    opt_s = torch.optim.AdamW(probe_small.parameters(), lr=LR_S, weight_decay=1e-3)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=EPOCHS_S, eta_min=1e-6)
    ce_s  = nn.CrossEntropyLoss(label_smoothing=0.05)
    tr_idx_s = s_tr["pos_idx"].values
    tr_lab_s = torch.tensor(s_tr["seen_id"].values, dtype=torch.long)
    best_acc_s = 0.
    for epoch in range(EPOCHS_S):
        probe_small.train()
        perm = np.random.permutation(len(tr_idx_s))
        for i in range(0, len(perm), BATCH_S):
            idx = tr_idx_s[perm[i:i+BATCH_S]]
            v   = dino_emb_small[idx].to(DEVICE)
            l   = tr_lab_s[perm[i:i+BATCH_S]].to(DEVICE)
            opt_s.zero_grad()
            ce_s(probe_small(v), l).backward()
            opt_s.step()
        sch_s.step()
        if (epoch + 1) % 20 == 0:
            probe_small.eval()
            with torch.no_grad():
                vv   = dino_emb_small[s_val["pos_idx"].values].to(DEVICE)
                pv   = [seen_classes[i] for i in probe_small(vv).argmax(1).cpu().numpy()]
                vacc = pacc(pv, v_labels_small)
            if vacc > best_acc_s:
                best_acc_s = vacc
                torch.save(probe_small.state_dict(), SMALL_PROBE_BEST)
            print(f"  Epoch {epoch+1:3d}/{EPOCHS_S}  val={vacc*100:.2f}%")
    probe_small.load_state_dict(
        torch.load(SMALL_PROBE_BEST, map_location=DEVICE, weights_only=False))
    probe_small.eval()
    print(f"  Best val acc (small): {best_acc_s*100:.2f}%")

def load_clip_cache(path):
    if os.path.exists(path):
        d = torch.load(path, weights_only=False)
        return d["feat"], np.array(d["labels"])
    return None, None

clip_s_feat, clip_s_lab = load_clip_cache(os.path.join(SAVE_DIR, "clip_features_seen_test.pt"))
clip_v_feat, clip_v_lab = load_clip_cache(os.path.join(SAVE_DIR, "clip_features_val.pt"))
clip_u_feat, clip_u_lab = load_clip_cache(os.path.join(SAVE_DIR, "clip_features_unseen.pt"))

if clip_s_feat is None:
    print("  WARNING: CLIP caches not found. Run main_bcvsa.py first.")
else:
    clip_s_raw = (clip_s_feat @ all_text_cpu.T).numpy()
    clip_v_raw = (clip_v_feat @ all_text_cpu.T).numpy()
    clip_u_raw = (clip_u_feat @ all_text_cpu.T).numpy()

    np.random.seed(42)
    u_meta_arr = np.arange(len(clip_u_lab))
    uval_mask  = np.zeros(len(clip_u_lab), dtype=bool)
    utest_mask = np.zeros(len(clip_u_lab), dtype=bool)
    for cls in unseen_classes:
        pos  = u_meta_arr[clip_u_lab == cls]
        perm = np.random.permutation(len(pos))
        half = len(pos) // 2
        uval_mask[pos[perm[:half]]]  = True
        utest_mask[pos[perm[half:]]] = True

    u_labels_val_b  = clip_u_lab[uval_mask]
    u_labels_test_b = clip_u_lab[utest_mask]
    clip_u_raw_val  = clip_u_raw[uval_mask]
    clip_u_raw_test = clip_u_raw[utest_mask]

    print("  Running calibration grid for small backbone...")
    T_CLIPS  = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
    ALPHAS   = np.round(np.linspace(0, 1, 11), 2)
    COARSE_G = np.linspace(-0.5, 3.0, 351, dtype=np.float32)

    probe_sm_v_s   = get_probe_sm(dino_emb_small, s_val["pos_idx"].values,  probe_small)
    probe_sm_s_s   = get_probe_sm(dino_emb_small, s_test["pos_idx"].values, probe_small)
    probe_full_v_s = expand_full(probe_sm_v_s, len(v_labels_small), C, seen_ids_np)
    probe_full_s_s = expand_full(probe_sm_s_s, len(s_labels_small),  C, seen_ids_np)

    best_s = {"H": -1., "alpha": 0.4, "gamma": 0.05, "T": 0.05}
    for T_clip in tqdm(T_CLIPS, desc="  Calibrating"):
        cv_sc = softmax_np(clip_v_raw,     T_clip)
        cu_sc = softmax_np(clip_u_raw_val, T_clip)
        for alpha in ALPHAS:
            n_use   = min(len(cv_sc), len(probe_full_v_s))
            fv_base = alpha * probe_full_v_s[:n_use] + (1 - alpha) * cv_sc[:n_use]
            fu_base = (1 - alpha) * cu_sc
            y_sv    = clip_v_lab[:n_use]
            bH, bg  = -1., 0.
            for g in COARSE_G:
                fv = fv_base.copy(); fu = fu_base.copy()
                fv[:, seen_ids_np] -= g
                fu[:, seen_ids_np] -= g
                H = hm(pacc(all_cls_arr[fv.argmax(1)], y_sv),
                        pacc(all_cls_arr[fu.argmax(1)], u_labels_val_b))
                if H > bH:
                    bH, bg = H, g
            for g in np.linspace(max(-0.5, bg-0.25), bg+0.25, 201, dtype=np.float32):
                fv = fv_base.copy(); fu = fu_base.copy()
                fv[:, seen_ids_np] -= g
                fu[:, seen_ids_np] -= g
                H = hm(pacc(all_cls_arr[fv.argmax(1)], y_sv),
                        pacc(all_cls_arr[fu.argmax(1)], u_labels_val_b))
                if H > best_s["H"]:
                    best_s.update({"H": H, "alpha": float(alpha),
                                   "gamma": float(g), "T": T_clip})

    af_s = best_s["alpha"]
    gf_s = best_s["gamma"]
    Tf_s = best_s["T"]
    print(f"  Small optimal: alpha={af_s}  gamma={gf_s:.4f}  T={Tf_s}")
    print(f"  Small val H   = {best_s['H']*100:.2f}%")

    n_use    = min(len(probe_full_s_s), len(clip_s_raw))
    cs_sc_s  = softmax_np(clip_s_raw[:n_use], Tf_s)
    cu_sc_s  = softmax_np(clip_u_raw_test,    Tf_s)
    fs_s     = af_s * probe_full_s_s[:n_use] + (1 - af_s) * cs_sc_s
    fu_s     = (1 - af_s) * cu_sc_s
    fs_s[:, seen_ids_np] -= gf_s
    fu_s[:, seen_ids_np] -= gf_s
    s_lab_s_eval = s_labels_small[:n_use]
    _, _, S_small, U_small, H_small = gzsl_eval(
        fs_s, fu_s, s_lab_s_eval, u_labels_test_b, all_cls_arr)
    f1_s_small = f1_score(s_lab_s_eval, all_cls_arr[fs_s.argmax(1)],
                           average="macro", labels=seen_classes, zero_division=0)
    f1_u_small = f1_score(u_labels_test_b, all_cls_arr[fu_s.argmax(1)],
                           average="macro", labels=unseen_classes, zero_division=0)
    TEMP_ZSL  = 0.01
    zsl_small = pacc(
        unseen_arr[softmax_np(clip_u_raw[:, unseen_ids_np], TEMP_ZSL).argmax(1)],
        clip_u_lab)
    print(f"\n  SMALL: ZSL={zsl_small*100:.2f}%  "
          f"S={S_small*100:.2f}%  U={U_small*100:.2f}%  H={H_small*100:.2f}%")


    results_csv = os.path.join(RESULTS_DIR, "results_summary.csv")
    large_found = False
    if os.path.exists(results_csv):
        df_res    = pd.read_csv(results_csv)
        large_row = df_res[df_res["Method"].str.contains("Proposed.*0-shot", na=False)]
        if len(large_row):
            S_large   = float(large_row["S%"].values[0])   / 100
            U_large   = float(large_row["U%"].values[0])   / 100
            H_large   = float(large_row["H%"].values[0])   / 100
            ZSL_large = float(large_row["ZSL%"].values[0]) / 100
            F1S_large = float(large_row["F1_Seen%"].values[0])   / 100
            F1U_large = float(large_row["F1_Unseen%"].values[0]) / 100
            large_found = True

    if not large_found:
        print(f"  WARNING: could not find the 'BCVSA (Proposed) 0-shot' row in "
              f"{results_csv}.")
        print("  Run main_bcvsa.py first so this comparison uses your actual "
              "current numbers. Skipping the Large-vs-Small comparison table "
              "and figure rather than reporting stale placeholder values.")
    else:
        comparison = pd.DataFrame([
            {"Backbone": "DINOv2-Large (1024-dim, 307M params)",
             "ZSL%": round(ZSL_large*100, 2), "S%": round(S_large*100, 2),
             "U%": round(U_large*100, 2),     "H%": round(H_large*100, 2),
             "F1_Seen%": round(F1S_large*100, 2), "F1_Unseen%": round(F1U_large*100, 2)},
            {"Backbone": "DINOv2-Small (384-dim, 22M params)",
             "ZSL%": round(zsl_small*100, 2), "S%": round(S_small*100, 2),
             "U%": round(U_small*100, 2),     "H%": round(H_small*100, 2),
             "F1_Seen%": round(f1_s_small*100, 2), "F1_Unseen%": round(f1_u_small*100, 2)},
        ])
        comparison.to_csv(os.path.join(RESULTS_DIR, "backbone_comparison.csv"), index=False)
        print("\n  Backbone comparison:")
        print(comparison.to_string(index=False))

        metrics    = ["ZSL%", "S%", "U%", "H%", "F1-Seen%", "F1-Unseen%"]
        large_vals = [ZSL_large*100, S_large*100, U_large*100,
                      H_large*100, F1S_large*100, F1U_large*100]
        small_vals = [zsl_small*100, S_small*100, U_small*100,
                      H_small*100, f1_s_small*100, f1_u_small*100]
        x = np.arange(len(metrics))
        w = 0.32
        fig_bb, ax_bb = plt.subplots(figsize=(11, 5))
        bars_l = ax_bb.bar(x - w/2, large_vals, w, label="DINOv2-Large (307M, 1024-dim)",
                            color="#2563EB", alpha=0.88, edgecolor="white")
        bars_s = ax_bb.bar(x + w/2, small_vals, w, label="DINOv2-Small (22M, 384-dim)",
                            color="#F97316", alpha=0.88, edgecolor="white")
        for bars in [bars_l, bars_s]:
            for rect in bars:
                v = rect.get_height()
                ax_bb.text(rect.get_x() + rect.get_width()/2, v + 0.5,
                           f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax_bb.set_xticks(x)
        ax_bb.set_xticklabels(metrics, fontsize=10)
        ax_bb.set_ylim(0, 105)
        ax_bb.set_ylabel("Score (%)", fontsize=11)
        ax_bb.set_title(
            "Backbone Robustness: DINOv2-Large vs DINOv2-Small\n"
            "(Same BCVSA framework, same CLIP prototypes, independent calibration per backbone)",
            fontweight="bold", fontsize=11)
    

        ax_bb.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.16),
                     ncol=2, frameon=False)
        ax_bb.grid(alpha=0.3, axis="y")
        H_delta = H_large*100 - H_small*100
        h_idx   = metrics.index("H%")
        ax_bb.annotate(
            f"\u0394H = {H_delta:+.1f} pp",
            xy=(x[h_idx], max(large_vals[h_idx], small_vals[h_idx]) + 6),
            ha="center", va="bottom", fontsize=9.5, color="#DC2626", fontweight="bold")
        plt.tight_layout()
        savefig(fig_bb, "figure11_backbone_comparison.png")


section("Summary")
print(f"  Total runtime : {elapsed()}")
print(f"  Results dir   : {RESULTS_DIR}")
print()
outputs = [
    ("figure10_xai_attention_final.png", RESULTS_DIR),
    ("figure11_backbone_comparison.png", RESULTS_DIR),
    ("backbone_comparison.csv",          RESULTS_DIR),
    ("PLANT_embeddings_small.npy",       SAVE_DIR),
    ("probe_small_best.pt",              SAVE_DIR),
]
for fname, fdir in outputs:
    status = "✓" if os.path.exists(os.path.join(fdir, fname)) else "✗ not found"
    print(f"  {status}  {fname}")
print()
if clip_s_feat is not None and 'H_small' in dir() and 'large_found' in dir() and large_found:
    print(f"  DINOv2-Large  H = {H_large*100:.2f}%  (307M params)")
    print(f"  DINOv2-Small  H = {H_small*100:.2f}%  (22M  params)")
    print(f"  \u0394H              = {(H_large-H_small)*100:.2f} pp")
    print(f"  Param ratio     = {307//22}\u00d7 fewer parameters")