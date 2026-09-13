import os
import glob
import json
import pickle
import time
import copy
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
try:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(GLOBAL_SEED)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

t0 = time.time()


def elapsed():
    return f"{(time.time() - t0) / 60:.1f} min"


def section(msg):
    print(f"\n{'='*70}\n{msg}  ({elapsed()})\n{'='*70}")


def tick(msg):
    print(f"  [{elapsed()}] {msg}")


DATA_ROOT       = r"E:\VS Code\Projects\Thesis\TP002\PlantF\Data"
BUNDLE_PATH     = os.path.join(DATA_ROOT, "metadata", "PLANT_bundle.pkl")
SAVE_DIR        = os.path.join(DATA_ROOT, "metadata")
RESULTS_DIR     = os.path.join(DATA_ROOT, "metadata", "results")
TRAIN_DIR       = os.path.join(DATA_ROOT, "global", "train")
VAL_DIR         = os.path.join(DATA_ROOT, "global", "val")
SEEN_TEST_DIR   = os.path.join(DATA_ROOT, "global", "test")
UNSEEN_TEST_DIR = os.path.join(DATA_ROOT, "zero_shot", "unseen_test")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else
         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device: {DEVICE}")

CLIP_BATCH        = 64
DINO_BATCH        = 512
EPOCHS_PROBE      = 200
LR_PROBE          = 3e-4
TEMP_ZSL          = 0.01
UNSEEN_SPLIT_SEED = 42

RUN_MULTI_SEED = True
MULTI_SEEDS    = [42, 123, 456, 789, 1000]

ALIGN_EPOCHS   = 100
ALIGN_LR       = 3e-4
ALIGN_T        = 0.03
ALIGN_PATIENCE = 12

FL_ROUNDS    = 150
LOCAL_EPOCHS = 30
LOCAL_LR     = 3e-4
LOCAL_BATCH  = 256

CLIENTS = {
    "Bangladesh": ["Potato", "Tomato"],
    "India":      ["Tomato", "Grape", "Apple"],
    "USA":        ["Corn",   "Apple", "Grape", "Potato"],
    "Spain":      ["Grape"],
}

# DP-FedAvg (McMahan et al. 2018). Mechanism only -- no (epsilon, delta)
# accountant is run; see Section 13c print for the honest scope note.
RUN_DP_FL           = True
DP_CLIP_NORM        = 1.0
DP_NOISE_MULTIPLIER = 0.001

saved_figs = []


def savefig(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved_figs.append(path)
    print(f"  Saved: {fname}")


def softmax_np(x, T=1.0):
    x = np.asarray(x, dtype=np.float64) / T
    e = np.exp(x - x.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def hm(s, u):
    return 2 * s * u / (s + u + 1e-9)


def pacc(p, y):
    return float((np.asarray(p) == np.asarray(y)).mean())


def compute_ece(probs, y_true, label_arr, n_bins=15):
    conf = probs.max(1)
    pred = label_arr[probs.argmax(1)]
    corr = (pred == np.asarray(y_true)).astype(float)
    bins  = np.linspace(0, 1, n_bins + 1)
    total = 0.
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum():
            total += abs(corr[m].mean() - conf[m].mean()) * m.sum()
    return total / max(len(y_true), 1)


def full_metrics(y_true, y_pred, labels):
    return {
        "acc":  pacc(y_pred, y_true),
        "f1":   f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "prec": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "rec":  recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


def expand_full(sm, N, C, seen_ids):
    full = np.zeros((N, C), dtype=np.float32)
    for j, si in enumerate(seen_ids):
        full[:, si] = sm[:, j]
    return full


def gzsl_eval(fs, fu, y_s, y_u, cls_arr):
    pred_s = cls_arr[fs.argmax(1)]
    pred_u = cls_arr[fu.argmax(1)]
    S = pacc(pred_s, y_s)
    U = pacc(pred_u, y_u)
    return pred_s, pred_u, S, U, hm(S, U)


def gamma_sweep(sm_sv, sm_uv, seen_ids, cls_arr, y_sv, y_uv,
                gammas=None, n_fine=201):
    if gammas is None:
        gammas = np.linspace(-0.5, 3.0, 351, dtype=np.float32)
    best = {"H": -1., "g": 0., "S": 0., "U": 0.}

    def _ev(g):
        ts = sm_sv.copy()
        tu = sm_uv.copy()
        ts[:, seen_ids] -= g
        tu[:, seen_ids] -= g
        S = pacc(cls_arr[ts.argmax(1)], y_sv)
        U = pacc(cls_arr[tu.argmax(1)], y_uv)
        H = hm(S, U)
        if H > best["H"]:
            best.update({"H": H, "g": g, "S": S, "U": U})

    for g in gammas:
        _ev(g)
    lo = best["g"] - 0.2
    hi = best["g"] + 0.2
    for g in np.linspace(lo, hi, n_fine):
        _ev(g)
    return best["g"], best["H"], best["S"], best["U"]


def error_analysis(pred_arr, true_arr, class_list):
    rows = []
    for cls in class_list:
        mask = np.asarray(true_arr) == cls
        if not mask.sum():
            continue
        n      = int(mask.sum())
        preds  = np.asarray(pred_arr)[mask]
        correct = int((preds == cls).sum())
        wrong_p = preds[preds != cls]
        if len(wrong_p):
            vals, cnts = np.unique(wrong_p, return_counts=True)
            tw   = vals[cnts.argmax()]
            tw_n = int(cnts.max())
        else:
            tw   = "None"
            tw_n = 0
        pred_all = np.asarray(pred_arr)
        true_all = np.asarray(true_arr)
        tp = int(((pred_all == cls) & (true_all == cls)).sum())
        fp = int(((pred_all == cls) & (true_all != cls)).sum())
        fn = int(((pred_all != cls) & (true_all == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.
        rec  = tp / (tp + fn) if (tp + fn) else 0.
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.
        rows.append({
            "Class":       cls,
            "N":           n,
            "Correct":     correct,
            "Acc_%":       round(correct / n * 100, 2),
            "F1_%":        round(f1 * 100, 2),
            "Top_wrong":   tw,
            "Top_wrong_N": tw_n,
            "Top_wrong_%": round(tw_n / n * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("Acc_%")


def hd_analysis(pred_arr, true_arr, class_list, split):
    rows = []
    for cls in class_list:
        mask = np.asarray(true_arr) == cls
        if not mask.sum():
            continue
        n      = int(mask.sum())
        p      = np.asarray(pred_arr)[mask]
        correct = int((p == cls).sum())
        as_h   = int(sum("Healthy" in x for x in p))
        rows.append({
            "Split":          split,
            "Class":          cls,
            "Type":           "Healthy" if "Healthy" in cls else "Diseased",
            "N":              n,
            "Correct_%":      round(correct / n * 100, 2),
            "As_Healthy_%":   round(as_h / n * 100, 2),
            "As_Diseased_%":  round((n - as_h) / n * 100, 2),
        })
    return pd.DataFrame(rows)


section("[1] Loading Bundle")

with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

meta = pd.DataFrame(bundle["metadata"])
meta.index    = range(len(meta))
meta["pos_idx"] = meta.index

if "class_name" not in meta.columns and "class" in meta.columns:
    meta = meta.rename(columns={"class": "class_name"})
if "type" not in meta.columns:
    meta["type"] = "seen"
    meta.loc[meta["split"].str.contains("unseen", case=False, na=False), "type"] = "unseen"

dino_emb   = F.normalize(torch.tensor(bundle["embeddings"], dtype=torch.float32), dim=-1)
train_meta = meta[(meta["type"] == "seen") & (meta["split"] == "train")].copy()
val_meta   = meta[(meta["type"] == "seen") & (meta["split"] == "val")].copy()
stest_meta = meta[(meta["type"] == "seen") & (meta["split"] == "test")].copy()
utest_meta = meta[meta["type"] == "unseen"].copy()

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
train_meta["seen_id"] = train_meta["class_name"].map(seen_local)
val_meta["seen_id"]   = val_meta["class_name"].map(seen_local)
print(f"  Seen={S_n}  Unseen={len(unseen_classes)}  All={C}  Total images={len(meta)}")


section("[2] Unseen 50/50 Calibration Split (seed=42)")

np.random.seed(UNSEEN_SPLIT_SEED)
u_meta_arr = np.arange(len(utest_meta))
u_cls_arr  = np.array(utest_meta["class_name"].tolist())
uval_mask  = np.zeros(len(utest_meta), dtype=bool)
utest_mask = np.zeros(len(utest_meta), dtype=bool)

for cls in unseen_classes:
    pos  = u_meta_arr[u_cls_arr == cls]
    perm = np.random.permutation(len(pos))
    half = len(pos) // 2
    uval_mask[pos[perm[:half]]]  = True
    utest_mask[pos[perm[half:]]] = True

N_uval  = int(uval_mask.sum())
N_utest = int(utest_mask.sum())
print(f"  Calibration set (50%) : {N_uval} images")
print(f"  Evaluation set  (50%) : {N_utest} images")


section("[3] CLIP Text Prototypes")

CROP_DESC = {
    "Apple":      "broad oval waxy leaf with serrated margins, apple tree",
    "Corn":       "long narrow blade monocot leaf with parallel veins, maize",
    "Tomato":     "compound pinnate leaf with serrated leaflets and hairy texture",
    "Grape":      "palmate lobed leaf with 5 lobes and prominent veins, grapevine",
    "Potato":     "pinnate compound leaf with dark green oval leaflets",
    "Strawberry": "trifoliate leaf three leaflets deeply serrated bright green",
    "Orange":     "single elliptical glossy leaf with winged petiole, citrus orange",
    "Peach":      "lanceolate narrow leaf finely serrated margins, stone fruit peach",
    "Blueberry":  "small oval smooth waxy leaf slightly bluish-green surface, vaccinium blueberry",
    "Raspberry":  "pinnate compound leaf pale silver underside thorny cane, rubus raspberry",
    "Squash":     "large palmate lobed rough hairy leaf, cucurbit squash zucchini",
    "Cherry":     "ovate leaf doubly serrated shiny surface, cherry prunus tree",
    "Soybean":    "trifoliate oval leaflets hairy surface, glycine max soybean legume",
    "Cassava":    "deeply palmate star-shaped lobed leaf long petiole, manihot cassava manioc",
    "Pepper_Bell":"smooth ovate leaf prominent midrib wavy margins, capsicum bell pepper",
}


def make_prompts(cls):
    parts  = cls.split("___")
    crop_k = parts[0]
    crop   = crop_k.replace("_", " ").lower()
    cond   = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
    desc   = CROP_DESC.get(crop_k, "")
    clean  = cls.replace("___", " ").replace("_", " ")
    is_h   = "healthy" in cond.lower()
    if is_h:
        return [
            f"a high-resolution photo of a healthy {clean} plant leaf",
            f"a healthy {clean} leaf in natural lighting",
            f"a close-up image of a disease-free {clean} leaf",
            f"a pristine leaf of {clean}",
            f"healthy {crop} leaf, {desc}",
            f"close-up photo of healthy {crop} plant leaf",
            f"vibrant normal {crop} leaf without symptoms, {desc}",
            f"undamaged {crop} foliage in natural condition",
            f"high resolution photo of {crop} plant leaf",
            f"a well-cared {clean} leaf with no disease",
        ]
    return [
        f"a high-resolution photo of a {clean} diseased plant leaf",
        f"a close-up image of {clean} disease symptoms",
        f"a plant leaf affected by {clean}",
        f"a leaf exhibiting signs of {clean} disease",
        f"a clear image of a {clean} infected leaf",
        f"high resolution photo of {crop} plant leaf",
        f"close-up photo of {crop} plant leaf showing {cond}",
        f"{crop} plant leaf with {cond} disease, {desc}",
        f"pathology image of {crop} {cond}",
        f"infected {crop} foliage with {cond} symptoms",
    ]


clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").float().to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
IPEX = False
try:
    import intel_extension_for_pytorch as ipex
    clip_model = ipex.optimize(clip_model)
    IPEX = True
    print("  IPEX enabled")
except Exception:
    print("  IPEX not available")

text_cache = os.path.join(SAVE_DIR, "clip_text_prototypes.pt")
if os.path.exists(text_cache):
    all_text_cpu = torch.load(text_cache, weights_only=False)
    print("  Loaded from cache")
else:
    text_features = {}
    with torch.no_grad():
        for cls in tqdm(all_classes, desc="  Building text prototypes"):
            prompts = make_prompts(cls)
            inp = clip_processor(text=prompts, return_tensors="pt",
                                 padding=True, truncation=True).to(DEVICE)
            f   = F.normalize(clip_model.get_text_features(**inp).float(), dim=-1)
            text_features[cls] = F.normalize(f.mean(0), dim=-1).cpu()
    all_text_cpu = torch.stack([text_features[c] for c in all_classes])
    torch.save(all_text_cpu, text_cache)

all_text_np = all_text_cpu.numpy()
print(f"  Text prototypes ready. Shape: {all_text_cpu.shape}")


section("[4] CLIP Image Feature Extraction")


def extract_clip(base_dir, cls_list, cache_path, label):
    if os.path.exists(cache_path):
        tick(f"Cache hit: {os.path.basename(cache_path)}")
        d = torch.load(cache_path, weights_only=False)
        return d["feat"], np.array(d["labels"]), d["class_filenames"]
    all_f, all_l, all_cf = [], [], []
    for cls in tqdm(cls_list, desc=f"  {label}"):
        cdir  = os.path.join(base_dir, cls)
        if not os.path.isdir(cdir):
            continue
        paths = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
        for i in range(0, len(paths), CLIP_BATCH):
            bp = paths[i:i + CLIP_BATCH]
            imgs = []
            for p in bp:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            inp = clip_processor(images=imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                feat = F.normalize(clip_model.get_image_features(**inp).float(), dim=-1)
            all_f.append(feat.cpu())
            all_l.extend([cls] * len(bp))
            all_cf.extend([f"{cls}/{os.path.basename(p)}" for p in bp])
            if DEVICE.type == "xpu":
                torch.xpu.empty_cache()
    fc = torch.cat(all_f)
    torch.save({"feat": fc, "labels": np.array(all_l), "class_filenames": all_cf}, cache_path)
    return fc, np.array(all_l), all_cf


clip_u_all, u_lab_all, u_cf_all = extract_clip(
    UNSEEN_TEST_DIR, unseen_classes,
    os.path.join(SAVE_DIR, "clip_features_unseen.pt"), "Unseen")
clip_v, v_labels, _ = extract_clip(
    VAL_DIR, seen_classes,
    os.path.join(SAVE_DIR, "clip_features_val.pt"), "Val")
clip_s, s_labels, _ = extract_clip(
    SEEN_TEST_DIR, seen_classes,
    os.path.join(SAVE_DIR, "clip_features_seen_test.pt"), "Seen-Test")
del clip_model

u_cf_to_idx  = {cf: i for i, cf in enumerate(u_cf_all)}
u_lab_all_np = np.array(u_lab_all)
clip_u_val   = clip_u_all[uval_mask]
clip_u_test  = clip_u_all[utest_mask]
u_labels_val  = u_lab_all_np[uval_mask]
u_labels_test = u_lab_all_np[utest_mask]
s_labels_np   = np.array(s_labels)
v_labels_np   = np.array(v_labels)

clip_u_raw_all  = (clip_u_all  @ all_text_cpu.T).numpy()
clip_u_raw_val  = (clip_u_val  @ all_text_cpu.T).numpy()
clip_u_raw_test = (clip_u_test @ all_text_cpu.T).numpy()
clip_s_raw      = (clip_s      @ all_text_cpu.T).numpy()
clip_v_raw      = (clip_v      @ all_text_cpu.T).numpy()
tick("CLIP cosine similarities computed")


section(f"[5] DINOv2 Linear Probe Training ({EPOCHS_PROBE} epochs)")


class LinearProbe(nn.Module):
    def __init__(self, out=S_n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512,  256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out))

    def forward(self, x):
        return self.net(x)


probe_path_best  = os.path.join(SAVE_DIR, "probe_best.pt")
probe_path_final = os.path.join(SAVE_DIR, "probe_final.pt")

probe  = LinearProbe().to(DEVICE)
opt_p  = torch.optim.AdamW(probe.parameters(), lr=LR_PROBE, weight_decay=1e-3)
sch_p  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=EPOCHS_PROBE, eta_min=1e-6)
ce_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)
if IPEX:
    try:
        probe, opt_p = ipex.optimize(probe, optimizer=opt_p, level="O1")
    except Exception:
        pass

tr_idx   = train_meta["pos_idx"].values
tr_lab   = torch.tensor(train_meta["seen_id"].values, dtype=torch.long)
best_acc = 0.

for epoch in range(EPOCHS_PROBE):
    probe.train()
    perm = np.random.permutation(len(tr_idx))
    for i in range(0, len(perm), DINO_BATCH):
        idx = tr_idx[perm[i:i + DINO_BATCH]]
        v   = dino_emb[idx].to(DEVICE)
        l   = tr_lab[perm[i:i + DINO_BATCH]].to(DEVICE)
        opt_p.zero_grad()
        ce_fn(probe(v), l).backward()
        opt_p.step()
    sch_p.step()
    if (epoch + 1) % 20 == 0:
        probe.eval()
        with torch.no_grad():
            vv   = dino_emb[val_meta["pos_idx"].values].to(DEVICE)
            pv   = [seen_classes[i] for i in probe(vv).argmax(1).cpu().numpy()]
            vacc = accuracy_score(val_meta["class_name"].tolist(), pv)
        if vacc > best_acc:
            best_acc = vacc
            torch.save(probe.state_dict(), probe_path_best)
        print(f"  Epoch {epoch+1:3d}/{EPOCHS_PROBE}  val={vacc*100:.2f}%  ({elapsed()})")

probe.load_state_dict(torch.load(probe_path_best, map_location=DEVICE, weights_only=False))
probe.eval()
torch.save(probe.state_dict(), probe_path_final)
print(f"  Best validation accuracy: {best_acc*100:.2f}%")


def get_probe_sm(idx_arr, model=None):
    m   = model if model else probe
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_arr), DINO_BATCH):
            idx = idx_arr[i:i + DINO_BATCH]
            out.append(F.softmax(m(dino_emb[idx].to(DEVICE)), dim=-1).cpu().numpy())
    return np.vstack(out)


def get_probe_logits(idx_arr, model=None):
    m   = model if model else probe
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_arr), DINO_BATCH):
            idx = idx_arr[i:i + DINO_BATCH]
            out.append(m(dino_emb[idx].to(DEVICE)).cpu().numpy())
    return np.vstack(out)


probe_sm_s   = get_probe_sm(stest_meta["pos_idx"].values)
probe_sm_v   = get_probe_sm(val_meta["pos_idx"].values)
probe_full_s = expand_full(probe_sm_s, len(s_labels), C, seen_ids_np)
probe_full_v = expand_full(probe_sm_v, len(v_labels), C, seen_ids_np)

probe_logit_s = expand_full(get_probe_logits(stest_meta["pos_idx"].values), len(s_labels), C, seen_ids_np)
probe_logit_v = expand_full(get_probe_logits(val_meta["pos_idx"].values),   len(v_labels), C, seen_ids_np)

pred_b  = [seen_classes[i] for i in probe_sm_s.argmax(1)]
met_b   = full_metrics(s_labels_np, pred_b, seen_classes)
print(f"  Ablation B (supervised): acc={met_b['acc']*100:.2f}%  F1={met_b['f1']*100:.2f}%")


section("[6] Calibration (val only — no test leakage)")

# T searched descending so a saturated softmax keeps the mildest T on a
# tie instead of whichever value is tried first.
T_CLIPS  = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
ALPHAS   = np.round(np.linspace(0, 1, 11), 2)
COARSE_G = np.linspace(-0.5, 3.0, 351, dtype=np.float32)


def run_calibration_search(probe_full_v_arg, desc="  T_clip"):
    best = {"H": -1., "alpha": 0.4, "gamma": 0.026, "T_clip": 0.07, "S_val": 0., "U_val": 0.}
    for T_clip in tqdm(sorted(T_CLIPS, reverse=True), desc=desc):
        cv_sc = softmax_np(clip_v_raw,     T_clip)
        cu_sc = softmax_np(clip_u_raw_val, T_clip)
        for alpha in ALPHAS:
            fv_base = alpha * probe_full_v_arg + (1 - alpha) * cv_sc
            fu_base = (1 - alpha) * cu_sc
            bH = -1.
            bg = 0.
            for g in COARSE_G:
                fv = fv_base.copy()
                fu = fu_base.copy()
                fv[:, seen_ids_np] -= g
                fu[:, seen_ids_np] -= g
                S  = pacc(all_cls_arr[fv.argmax(1)], v_labels_np)
                U  = pacc(all_cls_arr[fu.argmax(1)], u_labels_val)
                H  = hm(S, U)
                if H > bH:
                    bH, bg = H, g
            for g in np.linspace(max(-0.5, bg - 0.25), bg + 0.25, 201, dtype=np.float32):
                fv = fv_base.copy()
                fu = fu_base.copy()
                fv[:, seen_ids_np] -= g
                fu[:, seen_ids_np] -= g
                S  = pacc(all_cls_arr[fv.argmax(1)], v_labels_np)
                U  = pacc(all_cls_arr[fu.argmax(1)], u_labels_val)
                H  = hm(S, U)
                if H > best["H"]:
                    best.update({"H": H, "alpha": float(alpha),
                                 "gamma": float(g), "T_clip": T_clip,
                                 "S_val": S, "U_val": U})
    if best["T_clip"] == min(T_CLIPS):
        print(f"  WARNING: chosen T_clip={best['T_clip']} is still the "
              f"smallest value in T_CLIPS -- extend it downward and re-run.")
    return best


print("  Grid search: T_clip x alpha x gamma ...")
best_cal = run_calibration_search(probe_full_v)

af = best_cal["alpha"]
gf = best_cal["gamma"]
Tf = best_cal["T_clip"]
print(f"  Val H={best_cal['H']*100:.2f}%  S={best_cal['S_val']*100:.2f}%  U={best_cal['U_val']*100:.2f}%")
print(f"  Optimal params: alpha={af}  gamma={gf:.4f}  T={Tf}")

print(f"  Building calibration heatmap (T={Tf})...")
alpha_gamma_grid = {}
cv_sc_tf = softmax_np(clip_v_raw,     Tf)
cu_sc_tf = softmax_np(clip_u_raw_val, Tf)
gammas_hm = np.linspace(0.0, 0.5, 51, dtype=np.float32)
for alpha in ALPHAS:
    fv_b = alpha * probe_full_v + (1 - alpha) * cv_sc_tf
    fu_b = (1 - alpha) * cu_sc_tf
    for g in gammas_hm:
        fv = fv_b.copy()
        fu = fu_b.copy()
        fv[:, seen_ids_np] -= g
        fu[:, seen_ids_np] -= g
        S  = pacc(all_cls_arr[fv.argmax(1)], v_labels_np)
        U  = pacc(all_cls_arr[fu.argmax(1)], u_labels_val)
        alpha_gamma_grid[(round(float(alpha), 2), round(float(g), 4))] = hm(S, U)
print(f"  Heatmap grid: {len(alpha_gamma_grid)} points")

u_label_idx_val = np.array([cls2id[c] for c in u_labels_val])
v_label_idx     = np.array([cls2id[c] for c in v_labels_np])


def nll_unseen(log_T):
    T  = float(np.exp(log_T))
    sc = clip_u_raw_val.copy() / T
    sc[:, seen_ids_np] -= gf
    sh  = sc - sc.max(1, keepdims=True)
    lsm = np.log(np.exp(sh).sum(1, keepdims=True))
    return -(sh[np.arange(len(u_label_idx_val)), u_label_idx_val] - lsm.squeeze()).mean()


res_u        = minimize_scalar(nll_unseen, bounds=(-4., 2.), method="bounded",
                               options={"xatol": 1e-4})
T_unseen_cal = float(np.exp(res_u.x))

seen_logit_v = af * probe_logit_v + (1 - af) * (clip_v_raw / Tf)
seen_logit_v[:, seen_ids_np] -= gf


def nll_seen(log_T):
    T  = float(np.exp(log_T))
    sc = seen_logit_v.copy() / T
    sh  = sc - sc.max(1, keepdims=True)
    lsm = np.log(np.exp(sh).sum(1, keepdims=True))
    return -(sh[np.arange(len(v_label_idx)), v_label_idx] - lsm.squeeze()).mean()


res_s      = minimize_scalar(nll_seen, bounds=(-4., 2.), method="bounded",
                             options={"xatol": 1e-4})
T_seen_cal = float(np.exp(res_s.x))
print(f"  T_unseen={T_unseen_cal:.4f}  T_seen={T_seen_cal:.4f}  (both from val set)")


section("[7] GZSL Evaluation (held-out 50% unseen test set)")

cs_sc = softmax_np(clip_s_raw,      Tf)
cu_sc = softmax_np(clip_u_raw_test, Tf)
fs_p  = af * probe_full_s + (1 - af) * cs_sc
fs_p[:, seen_ids_np] -= gf
fu_p  = (1 - af) * cu_sc
fu_p[:, seen_ids_np] -= gf
pred_s_p, pred_u_p, acc_s_p, acc_u_p, H_p = gzsl_eval(
    fs_p, fu_p, s_labels_np, u_labels_test, all_cls_arr)

pred_zsl_p = unseen_arr[(clip_u_raw_all[:, unseen_ids_np] / TEMP_ZSL).argmax(1)]
acc_zsl_p  = pacc(pred_zsl_p, u_lab_all_np)

met_p_s   = full_metrics(s_labels_np,   pred_s_p,  seen_classes)
met_p_u   = full_metrics(u_labels_test, pred_u_p,  unseen_classes)
met_zsl_p = full_metrics(u_lab_all_np,  pred_zsl_p, unseen_classes)

fu_cal_logit = clip_u_raw_test.copy() / T_unseen_cal
fu_cal_logit[:, seen_ids_np] -= gf
fu_cal_exp   = np.exp(fu_cal_logit - fu_cal_logit.max(1, keepdims=True))
fu_cal       = fu_cal_exp / fu_cal_exp.sum(1, keepdims=True)

seen_logit_s = af * probe_logit_s + (1 - af) * (clip_s_raw / Tf)
seen_logit_s[:, seen_ids_np] -= gf
fs_logit     = seen_logit_s / T_seen_cal
fs_cal_exp   = np.exp(fs_logit - fs_logit.max(1, keepdims=True))
fs_cal       = fs_cal_exp / fs_cal_exp.sum(1, keepdims=True)
ece_p_s      = compute_ece(fs_cal,  s_labels_np,   all_cls_arr)
ece_p_u      = compute_ece(fu_cal,  u_labels_test, all_cls_arr)

print(f"  ZSL   (N={len(u_lab_all)}): {acc_zsl_p*100:.2f}%  F1={met_zsl_p['f1']*100:.2f}%")
print(f"  GZSL  0-shot: S={acc_s_p*100:.2f}%  U={acc_u_p*100:.2f}%  H={H_p*100:.2f}%")
print(f"  F1    Seen={met_p_s['f1']*100:.2f}%   Unseen={met_p_u['f1']*100:.2f}%")
print(f"  ECE   Seen={ece_p_s*100:.2f}%   Unseen={ece_p_u*100:.2f}%")


section("[8] Few-Shot GZSL (3-shot and 5-shot)")

clip_model2     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").float().to(DEVICE)
clip_processor2 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model2.eval()
if IPEX:
    try:
        clip_model2 = ipex.optimize(clip_model2)
    except Exception:
        pass


def few_shot_proposed(k, seed=42):
    np.random.seed(seed)
    atf = all_text_cpu.clone()
    support_global_indices = set()

    for cls in unseen_classes:
        paths = sorted(glob.glob(os.path.join(UNSEEN_TEST_DIR, cls, "*.jpg")))
        if len(paths) < k:
            continue
        chosen = list(np.random.choice(paths, k, replace=False))
        imgs   = [Image.open(p).convert("RGB") for p in chosen]
        inp    = clip_processor2(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vf = F.normalize(clip_model2.get_image_features(**inp).float(), dim=-1).cpu()
        atf[cls2id[cls]] = F.normalize(
            0.5 * vf.mean(0) + 0.5 * all_text_cpu[cls2id[cls]], dim=-1)
        for p in chosen:
            cf_key = f"{cls}/{os.path.basename(p)}"
            if cf_key in u_cf_to_idx:
                support_global_indices.add(u_cf_to_idx[cf_key])

    zsl_eval_mask = np.ones(len(u_lab_all), dtype=bool)
    if support_global_indices:
        zsl_eval_mask[np.array(sorted(support_global_indices))] = False
    cu_fs_all  = (clip_u_all @ atf.T).numpy()
    acc_zsl_fs = pacc(
        unseen_arr[(cu_fs_all[zsl_eval_mask][:, unseen_ids_np] / TEMP_ZSL).argmax(1)],
        u_lab_all_np[zsl_eval_mask])

    test_global_idx = np.where(utest_mask)[0]
    support_in_test = support_global_indices & set(test_global_idx.tolist())
    test_eval_mask  = np.ones(N_utest, dtype=bool)
    if support_in_test:
        global_to_local = {g: i for i, g in enumerate(test_global_idx)}
        for g in support_in_test:
            test_eval_mask[global_to_local[g]] = False

    cu_fs_test  = (clip_u_test @ atf.T).numpy()
    cu_sc_fs    = softmax_np(cu_fs_test, Tf)
    cs_sc_orig  = softmax_np(clip_s_raw, Tf)
    fs_fs       = af * probe_full_s + (1 - af) * cs_sc_orig
    fs_fs[:, seen_ids_np] -= gf
    fu_fs       = (1 - af) * cu_sc_fs
    fu_fs[:, seen_ids_np] -= gf

    pred_s_fs     = all_cls_arr[fs_fs.argmax(1)]
    pred_u_fs     = all_cls_arr[fu_fs[test_eval_mask].argmax(1)]
    u_labels_eval = u_labels_test[test_eval_mask]
    acc_s_fs = pacc(pred_s_fs, s_labels_np)
    acc_u_fs = pacc(pred_u_fs, u_labels_eval)
    H_fs     = hm(acc_s_fs, acc_u_fs)
    met_s_fs = full_metrics(s_labels_np,   pred_s_fs, seen_classes)
    met_u_fs = full_metrics(u_labels_eval, pred_u_fs, unseen_classes)
    return acc_zsl_fs, acc_s_fs, acc_u_fs, H_fs, met_s_fs, met_u_fs


zsl3_p, s3_p, u3_p, H3_p, ms3_p, mu3_p = few_shot_proposed(3)
zsl5_p, s5_p, u5_p, H5_p, ms5_p, mu5_p = few_shot_proposed(5)
print(f"  3-shot: ZSL={zsl3_p*100:.2f}%  S={s3_p*100:.2f}%  U={u3_p*100:.2f}%  H={H3_p*100:.2f}%")
print(f"  5-shot: ZSL={zsl5_p*100:.2f}%  S={s5_p*100:.2f}%  U={u5_p*100:.2f}%  H={H5_p*100:.2f}%")

multi_seed_H3 = [H3_p]
multi_seed_H5 = [H5_p]
if RUN_MULTI_SEED:
    print(f"\n  Multi-seed robustness check ({MULTI_SEEDS})...")
    for s in MULTI_SEEDS:
        if s == 42:
            continue
        r3 = few_shot_proposed(3, seed=s)
        r5 = few_shot_proposed(5, seed=s)
        multi_seed_H3.append(r3[3])
        multi_seed_H5.append(r5[3])
        print(f"    seed={s}  H3={r3[3]*100:.2f}%  H5={r5[3]*100:.2f}%")
    print(f"  3-shot H: {np.mean(multi_seed_H3)*100:.2f}% +/- {np.std(multi_seed_H3)*100:.2f}%")
    print(f"  5-shot H: {np.mean(multi_seed_H5)*100:.2f}% +/- {np.std(multi_seed_H5)*100:.2f}%")

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


section("[9] Ablation A — CLIP Only")

best_a = {"H": -1., "T": 0.15, "gamma": 0., "S": 0., "U": 0.}
for T in sorted(T_CLIPS, reverse=True):
    sm_sv_a = softmax_np(clip_v_raw,     T)
    sm_uv_a = softmax_np(clip_u_raw_val, T)
    g, H, S, U = gamma_sweep(sm_sv_a, sm_uv_a, seen_ids_np,
                              all_cls_arr, v_labels_np, u_labels_val)
    if H > best_a["H"]:
        best_a.update({"H": H, "T": T, "gamma": g, "S": S, "U": U})
if best_a["T"] == min(T_CLIPS):
    print(f"  WARNING: chosen Ablation-A T={best_a['T']} is still the "
          f"smallest value in T_CLIPS -- extend it downward and re-run.")

sm_as_test = softmax_np(clip_s_raw,      best_a["T"]).copy()
sm_au_test = softmax_np(clip_u_raw_test, best_a["T"]).copy()
sm_as_test[:, seen_ids_np] -= best_a["gamma"]
sm_au_test[:, seen_ids_np] -= best_a["gamma"]
pred_s_a, pred_u_a, acc_s_a, acc_u_a, H_a = gzsl_eval(
    sm_as_test, sm_au_test, s_labels_np, u_labels_test, all_cls_arr)
met_a_s = full_metrics(s_labels_np,   pred_s_a, seen_classes)
met_a_u = full_metrics(u_labels_test, pred_u_a, unseen_classes)

# T does not affect argmax (dividing a row by a positive constant never
# changes which class wins), so no search is needed or performed here.
# Evaluated on the unseen EVALUATION half only (utest_mask), matching
# Ablation C's protocol, rather than the full unseen set.
zsl_raw_a_test = clip_u_raw_all[utest_mask][:, unseen_ids_np]
best_zsl_a = pacc(unseen_arr[zsl_raw_a_test.argmax(1)], u_labels_test)

clip_v_logit_a = (clip_v_raw / best_a["T"]).copy()
clip_v_logit_a[:, seen_ids_np] -= best_a["gamma"]
v_label_idx_a = np.array([cls2id[c] for c in v_labels_np])


def _nll_a(log_T):
    T = float(np.exp(log_T))
    sc = clip_v_logit_a / T
    sh = sc - sc.max(1, keepdims=True)
    lsm = np.log(np.exp(sh).sum(1, keepdims=True))
    return -(sh[np.arange(len(v_label_idx_a)), v_label_idx_a] - lsm.squeeze()).mean()


res_a   = minimize_scalar(_nll_a, bounds=(-4., 2.), method="bounded", options={"xatol": 1e-4})
T_cal_a = float(np.exp(res_a.x))


def _calibrated_softmax_a(raw, mask_cols, T_base, gamma, T_cal):
    logit = (raw / T_base).copy()
    logit[:, mask_cols] -= gamma
    logit = logit / T_cal
    e = np.exp(logit - logit.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


sm_as_a_cal = _calibrated_softmax_a(clip_s_raw,      seen_ids_np, best_a["T"], best_a["gamma"], T_cal_a)
sm_au_a_cal = _calibrated_softmax_a(clip_u_raw_test, seen_ids_np, best_a["T"], best_a["gamma"], T_cal_a)
ece_a_s = compute_ece(sm_as_a_cal, s_labels_np,   all_cls_arr)
ece_a_u = compute_ece(sm_au_a_cal, u_labels_test, all_cls_arr)
print(f"  [Ablation A] ECE calibration temperature T_cal_a={T_cal_a:.4f}")
print(f"  ZSL={best_zsl_a*100:.2f}%  S={acc_s_a*100:.2f}%  U={acc_u_a*100:.2f}%  H={H_a*100:.2f}%")
print(f"  ECE Seen={ece_a_s*100:.2f}%  ECE Unseen={ece_a_u*100:.2f}%")


section("[10] Ablation C — Alignment Head (InfoNCE)")


class AlignmentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(2048, 768))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


align_head = AlignmentHead().to(DEVICE)
opt_al     = torch.optim.AdamW(align_head.parameters(), lr=ALIGN_LR, weight_decay=1e-2)
sched_al   = torch.optim.lr_scheduler.CosineAnnealingLR(opt_al, T_max=ALIGN_EPOCHS, eta_min=1e-5)
if IPEX:
    try:
        align_head, opt_al = ipex.optimize(align_head, optimizer=opt_al, level="O1")
    except Exception:
        pass

seen_tgt    = all_text_cpu[seen_ids_np]
best_proxy  = -1.
patience_c  = 0
best_al_w   = None
tr_meta_arr = train_meta["pos_idx"].values
tr_sid_arr  = train_meta["seen_id"].values

for epoch in range(ALIGN_EPOCHS):
    align_head.train()
    perm    = np.random.permutation(len(tr_meta_arr))
    ep_loss = 0.
    nb      = 0
    for i in range(0, len(perm), 128):
        bidx = tr_meta_arr[perm[i:i + 128]]
        bsid = tr_sid_arr[perm[i:i + 128]]
        if len(bidx) < 2:
            continue
        dv   = dino_emb[bidx].to(DEVICE)
        tt   = seen_tgt[bsid].to(DEVICE)
        p    = align_head(dv)
        loss = F.cross_entropy((p @ tt.T) / ALIGN_T, torch.arange(len(p), device=DEVICE))
        opt_al.zero_grad()
        loss.backward()
        opt_al.step()
        ep_loss += float(loss)
        nb += 1
    sched_al.step()
    if (epoch + 1) % 5 == 0:
        align_head.eval()
        with torch.no_grad():
            sidx  = val_meta["pos_idx"].values
            pv    = align_head(dino_emb[sidx].to(DEVICE)).cpu().numpy()
            proxy = pacc(all_cls_arr[(pv @ all_text_np.T).argmax(1)], v_labels_np)
        if proxy > best_proxy:
            best_proxy = proxy
            patience_c = 0
            best_al_w  = {k: v.cpu().clone() for k, v in align_head.state_dict().items()}
        else:
            patience_c += 1
        print(f"  Epoch {epoch+1:3d}  loss={ep_loss/max(nb,1):.4f}  "
              f"val_proxy={proxy*100:.1f}%  patience={patience_c}/{ALIGN_PATIENCE}")
        if patience_c >= ALIGN_PATIENCE:
            print("  Early stopping")
            break
        align_head.train()

if best_al_w:
    align_head.load_state_dict({k: v.to(DEVICE) for k, v in best_al_w.items()})
align_head.eval()


def project(idx_arr):
    parts = []
    with torch.no_grad():
        for i in range(0, len(idx_arr), DINO_BATCH):
            parts.append(
                align_head(dino_emb[idx_arr[i:i + DINO_BATCH]].to(DEVICE)).cpu().numpy())
    return np.vstack(parts)


proj_s      = project(stest_meta["pos_idx"].values)
proj_v      = project(val_meta["pos_idx"].values)
proj_u_all  = project(utest_meta["pos_idx"].values)
proj_u_val  = proj_u_all[uval_mask]
proj_u_test = proj_u_all[utest_mask]

aln_s_raw      = proj_s      @ all_text_np.T
aln_v_raw      = proj_v      @ all_text_np.T
aln_u_raw_val  = proj_u_val  @ all_text_np.T
aln_u_raw_test = proj_u_test @ all_text_np.T
tick("Alignment head projections done")

taus_c   = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.25, 1.5]
alphas_c = np.round(np.linspace(0., 1., 11), 2)
best_c   = {"H": -1., "tau": 0.05, "gamma": 0., "alpha": 1., "S": 0., "U": 0.}

for tau in tqdm(taus_c, desc="  Ablation C calibration"):
    sm_av = softmax_np(aln_v_raw,     tau)
    sm_cv = softmax_np(clip_v_raw,    tau)
    sm_au = softmax_np(aln_u_raw_val, tau)
    sm_cu = softmax_np(clip_u_raw_val, tau)
    for alpha in alphas_c:
        bl_v = alpha * sm_av + (1 - alpha) * sm_cv
        bl_u = alpha * sm_au + (1 - alpha) * sm_cu
        g, H, S, U = gamma_sweep(bl_v, bl_u, seen_ids_np,
                                  all_cls_arr, v_labels_np, u_labels_val)
        if H > best_c["H"]:
            best_c.update({"H": H, "tau": tau, "gamma": g,
                           "alpha": float(alpha), "S": S, "U": U})

tc = best_c["tau"]
gc = best_c["gamma"]
ac = best_c["alpha"]
print(f"  Ablation C val: H={best_c['H']*100:.2f}%  tau={tc}  gamma={gc:.4f}  alpha={ac}")

sm_as_c = ac * softmax_np(aln_s_raw,       tc) + (1 - ac) * softmax_np(clip_s_raw,       tc)
sm_au_c = ac * softmax_np(aln_u_raw_test,  tc) + (1 - ac) * softmax_np(clip_u_raw_test,  tc)
sm_as_c[:, seen_ids_np] -= gc
sm_au_c[:, seen_ids_np] -= gc
pred_s_c, pred_u_c, acc_s_c, acc_u_c, H_c = gzsl_eval(
    sm_as_c, sm_au_c, s_labels_np, u_labels_test, all_cls_arr)
met_c_s = full_metrics(s_labels_np,   pred_s_c, seen_classes)
met_c_u = full_metrics(u_labels_test, pred_u_c, unseen_classes)

aln_v_logit_c      = aln_v_raw  / tc
clip_v_logit_c      = clip_v_raw / tc
combined_v_logit_c  = ac * aln_v_logit_c + (1 - ac) * clip_v_logit_c
combined_v_logit_c[:, seen_ids_np] -= gc
v_label_idx_c = np.array([cls2id[c] for c in v_labels_np])


def _nll_c(log_T):
    T = float(np.exp(log_T))
    sc = combined_v_logit_c / T
    sh = sc - sc.max(1, keepdims=True)
    lsm = np.log(np.exp(sh).sum(1, keepdims=True))
    return -(sh[np.arange(len(v_label_idx_c)), v_label_idx_c] - lsm.squeeze()).mean()


res_c   = minimize_scalar(_nll_c, bounds=(-4., 2.), method="bounded", options={"xatol": 1e-4})
T_cal_c = float(np.exp(res_c.x))


def _calibrated_softmax_c(aln_raw, clip_raw, mask_cols, tc_, ac_, gamma, T_cal):
    logit = ac_ * (aln_raw / tc_) + (1 - ac_) * (clip_raw / tc_)
    logit[:, mask_cols] -= gamma
    logit = logit / T_cal
    e = np.exp(logit - logit.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


sm_as_c_cal = _calibrated_softmax_c(aln_s_raw,      clip_s_raw,      seen_ids_np, tc, ac, gc, T_cal_c)
sm_au_c_cal = _calibrated_softmax_c(aln_u_raw_test, clip_u_raw_test, seen_ids_np, tc, ac, gc, T_cal_c)
ece_c_s = compute_ece(sm_as_c_cal, s_labels_np,   all_cls_arr)
ece_c_u = compute_ece(sm_au_c_cal, u_labels_test, all_cls_arr)
print(f"  [Ablation C] ECE calibration temperature T_cal_c={T_cal_c:.4f}")

# ZSL (T, alpha) selected on the unseen CALIBRATION half only (uval_mask),
# then the resulting ZSL accuracy is reported on the unseen EVALUATION
# half only (utest_mask) -- previously both selection and reporting used
# the full unseen set, which is test leakage.
aln_u_raw_all = proj_u_all @ all_text_np.T
aln_zsl_all   = aln_u_raw_all[:, unseen_ids_np]
clip_zsl_all  = clip_u_raw_all[:, unseen_ids_np]

aln_zsl_val   = aln_zsl_all[uval_mask]
clip_zsl_val  = clip_zsl_all[uval_mask]
aln_zsl_eval  = aln_zsl_all[utest_mask]
clip_zsl_eval = clip_zsl_all[utest_mask]

best_zsl_val_c = 0.
best_zsl_p_c   = {"T": 0.05, "alpha": 0.}
for T in np.linspace(0.01, 0.5, 50):
    sm_az = softmax_np(aln_zsl_val, T)
    sm_cz = softmax_np(clip_zsl_val, T)
    for alp in alphas_c:
        a_z = pacc(unseen_arr[(alp * sm_az + (1 - alp) * sm_cz).argmax(1)], u_labels_val)
        if a_z > best_zsl_val_c:
            best_zsl_val_c = a_z
            best_zsl_p_c = {"T": T, "alpha": float(alp)}

Tz_c, alp_c = best_zsl_p_c["T"], best_zsl_p_c["alpha"]
sm_az_eval = softmax_np(aln_zsl_eval, Tz_c)
sm_cz_eval = softmax_np(clip_zsl_eval, Tz_c)
best_zsl_c = pacc(unseen_arr[(alp_c * sm_az_eval + (1 - alp_c) * sm_cz_eval).argmax(1)], u_labels_test)

print(f"  Ablation C test: ZSL={best_zsl_c*100:.2f}%  S={acc_s_c*100:.2f}%  "
      f"U={acc_u_c*100:.2f}%  H={H_c*100:.2f}%")
print(f"  Ablation C ECE:  Seen={ece_c_s*100:.2f}%  Unseen={ece_c_u*100:.2f}%")


def few_shot_align(k, seed=42):
    np.random.seed(seed)
    atf = all_text_cpu.clone()
    support_global_indices = set()

    for cls in unseen_classes:
        paths  = sorted(glob.glob(os.path.join(UNSEEN_TEST_DIR, cls, "*.jpg")))
        if len(paths) < k:
            continue
        chosen = list(np.random.choice(paths, k, replace=False))
        imgs   = [Image.open(p).convert("RGB") for p in chosen]
        inp    = clip_processor2(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vf = F.normalize(clip_model2.get_image_features(**inp).float(), dim=-1).cpu()
        atf[cls2id[cls]] = F.normalize(
            0.5 * vf.mean(0) + 0.5 * all_text_cpu[cls2id[cls]], dim=-1)
        for p in chosen:
            cf_key = f"{cls}/{os.path.basename(p)}"
            if cf_key in u_cf_to_idx:
                support_global_indices.add(u_cf_to_idx[cf_key])
    atn = atf.numpy()

    aln_s_fs       = proj_s      @ atn.T
    aln_v_fs       = proj_v      @ atn.T
    aln_u_fs_test  = proj_u_test @ atn.T
    aln_u_fs_val   = proj_u_val  @ atn.T
    cu_fs_test = (clip_u_test @ atf.T).numpy()
    cu_fs_val  = (clip_u_val  @ atf.T).numpy()
    cv_fs      = (clip_v      @ atf.T).numpy()
    cs_orig    = softmax_np(clip_s_raw, tc)

    bl_v  = ac * softmax_np(aln_v_fs,     tc) + (1 - ac) * softmax_np(cv_fs,     tc)
    bl_u  = ac * softmax_np(aln_u_fs_val, tc) + (1 - ac) * softmax_np(cu_fs_val, tc)
    g_fs, _, _, _ = gamma_sweep(bl_v, bl_u, seen_ids_np,
                                 all_cls_arr, v_labels_np, u_labels_val, n_fine=101)

    fs_fs = ac * softmax_np(aln_s_fs,      tc) + (1 - ac) * cs_orig
    fu_fs = ac * softmax_np(aln_u_fs_test, tc) + (1 - ac) * softmax_np(cu_fs_test, tc)
    fs_fs[:, seen_ids_np] -= g_fs
    fu_fs[:, seen_ids_np] -= g_fs

    test_global_idx = np.where(utest_mask)[0]
    support_in_test = support_global_indices & set(test_global_idx.tolist())
    test_eval_mask  = np.ones(N_utest, dtype=bool)
    if support_in_test:
        global_to_local = {g: i for i, g in enumerate(test_global_idx)}
        for g in support_in_test:
            test_eval_mask[global_to_local[g]] = False

    pred_s_fs     = all_cls_arr[fs_fs.argmax(1)]
    pred_u_fs     = all_cls_arr[fu_fs[test_eval_mask].argmax(1)]
    u_labels_eval = u_labels_test[test_eval_mask]
    S_fs = pacc(pred_s_fs, s_labels_np)
    U_fs = pacc(pred_u_fs, u_labels_eval)
    H_fs = hm(S_fs, U_fs)

    zsl_eval_mask = np.ones(len(u_lab_all), dtype=bool)
    if support_global_indices:
        zsl_eval_mask[np.array(sorted(support_global_indices))] = False
    cu_zsl_all  = (clip_u_all @ atf.T).numpy()
    aln_zsl_fs  = (proj_u_all @ atn.T)[:, unseen_ids_np]
    Tz    = best_zsl_p_c["T"]
    alp_z = best_zsl_p_c["alpha"]
    sm_zsl = (alp_z * softmax_np(aln_zsl_fs, Tz) +
              (1 - alp_z) * softmax_np(cu_zsl_all[:, unseen_ids_np], Tz))
    acc_zsl_fs = pacc(unseen_arr[sm_zsl[zsl_eval_mask].argmax(1)],
                      u_lab_all_np[zsl_eval_mask])

    ms = full_metrics(s_labels_np,   pred_s_fs, seen_classes)
    mu = full_metrics(u_labels_eval, pred_u_fs, unseen_classes)
    return acc_zsl_fs, S_fs, U_fs, H_fs, ms, mu, pred_s_fs, pred_u_fs


zsl3_c, s3_c, u3_c, H3_c, ms3_c, mu3_c, ps3_c, pu3_c = few_shot_align(3)
zsl5_c, s5_c, u5_c, H5_c, ms5_c, mu5_c, ps5_c, pu5_c = few_shot_align(5)
print(f"  Ablation C 3-shot: ZSL={zsl3_c*100:.2f}%  H={H3_c*100:.2f}%")
print(f"  Ablation C 5-shot: ZSL={zsl5_c*100:.2f}%  H={H5_c*100:.2f}%")

multi_seed_H3_c = [H3_c]
multi_seed_H5_c = [H5_c]
if RUN_MULTI_SEED:
    print(f"\n  [Ablation C] Multi-seed robustness check ({MULTI_SEEDS})...")
    for s in MULTI_SEEDS:
        if s == 42:
            continue
        r3c = few_shot_align(3, seed=s)
        r5c = few_shot_align(5, seed=s)
        multi_seed_H3_c.append(r3c[3])
        multi_seed_H5_c.append(r5c[3])
        print(f"    seed={s}  H3={r3c[3]*100:.2f}%  H5={r5c[3]*100:.2f}%")
    print(f"  [Ablation C] 3-shot H: {np.mean(multi_seed_H3_c)*100:.2f}% +/- {np.std(multi_seed_H3_c)*100:.2f}%")
    print(f"  [Ablation C] 5-shot H: {np.mean(multi_seed_H5_c)*100:.2f}% +/- {np.std(multi_seed_H5_c)*100:.2f}%")

del clip_model2

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


section("[11] Error Analysis")

df_err_p_u = error_analysis(pred_u_p, u_labels_test, unseen_classes)
df_err_p_s = error_analysis(pred_s_p, s_labels_np,   seen_classes)
df_hd_p_u  = hd_analysis(pred_u_p, u_labels_test, unseen_classes, "Unseen")
df_hd_p_s  = hd_analysis(pred_s_p, s_labels_np,   seen_classes,   "Seen")
df_err_c_u = error_analysis(pred_u_c, u_labels_test, unseen_classes)
df_err_c_s = error_analysis(pred_s_c, s_labels_np,   seen_classes)
df_hd_c_u  = hd_analysis(pred_u_c, u_labels_test, unseen_classes, "Unseen_AlignHead")
df_hd_c_s  = hd_analysis(pred_s_c, s_labels_np,   seen_classes,   "Seen_AlignHead")


def wrong_pairs(pred_arr, true_arr):
    rows = []
    for t, p in zip(true_arr, pred_arr):
        if t != p:
            rows.append({"True": t, "Predicted": p})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby(["True", "Predicted"]).size().reset_index(name="Count")
    df["True_s"] = df["True"].str.replace("___", " - ").str.replace("_", " ")
    df["Pred_s"] = df["Predicted"].str.replace("___", " - ").str.replace("_", " ")
    return df.sort_values("Count", ascending=False)


df_wp_p_u = wrong_pairs(pred_u_p, u_labels_test)
df_wp_p_s = wrong_pairs(pred_s_p, s_labels_np)
df_wp_c_u = wrong_pairs(pred_u_c, u_labels_test)

print("\n  Proposed — Unseen class accuracy (Acc% = pooled, F1% = per-class):")
print(f"  {'Class':<45} {'N':>6} {'Acc%':>7} {'F1%':>6} {'Top Confusion':<45} {'%':>6}")
for _, r in df_err_p_u.iterrows():
    cls_s = r["Class"].replace("___", " - ").replace("_", " ")[:43]
    tw_s  = str(r["Top_wrong"]).replace("___", " - ").replace("_", " ")[:43]
    print(f"  {cls_s:<45} {r['N']:>6} {r['Acc_%']:>6.1f}% {r['F1_%']:>5.1f}%  {tw_s:<45} {r['Top_wrong_%']:>5.1f}%")

h_avg = df_hd_p_u[df_hd_p_u["Type"] == "Healthy"]["Correct_%"].mean()
d_avg = df_hd_p_u[df_hd_p_u["Type"] == "Diseased"]["Correct_%"].mean()
print(f"\n  Healthy class avg: {h_avg:.1f}%   Diseased class avg: {d_avg:.1f}%")


section("[12] Severity Estimation")

crop_healthy_c  = {}
crop_diseased_c = {}
for cls in seen_classes:
    crop = cls.split("___")[0]
    idx  = train_meta[train_meta["class_name"] == cls]["pos_idx"].values
    if not len(idx):
        continue
    if "Healthy" in cls:
        crop_healthy_c[crop] = F.normalize(dino_emb[idx].mean(0), dim=-1)
    else:
        crop_diseased_c.setdefault(crop, []).append(dino_emb[idx])

for cls in unseen_classes:
    crop = cls.split("___")[0]
    cal_mask = (u_cls_arr == cls) & uval_mask
    idx = utest_meta["pos_idx"].values[cal_mask]
    if not len(idx):
        continue
    if "Healthy" in cls:
        if crop not in crop_healthy_c:
            crop_healthy_c[crop] = F.normalize(dino_emb[idx].mean(0), dim=-1)
    else:
        if crop not in crop_diseased_c:
            crop_diseased_c.setdefault(crop, []).append(dino_emb[idx])

crop_dis_centroid = {}
for crop, embs in crop_diseased_c.items():
    crop_dis_centroid[crop] = F.normalize(torch.cat(embs, 0).mean(0), dim=-1)

global_h = F.normalize(torch.stack(list(crop_healthy_c.values())).mean(0),  dim=-1)
global_d = F.normalize(torch.stack(list(crop_dis_centroid.values())).mean(0), dim=-1)


def compute_severity(emb_tensor, crop_list):
    crop_arr = np.array(crop_list)
    raw      = np.zeros(len(crop_list), dtype=np.float32)
    for crop in set(crop_list):
        mask = crop_arr == crop
        embs = emb_tensor[mask]
        h    = crop_healthy_c.get(crop, global_h)
        d    = crop_dis_centroid.get(crop, global_d)
        raw[mask] = (embs @ d).numpy() - (embs @ h).numpy()
    norm = np.zeros(len(crop_list), dtype=np.float32)
    for crop in set(crop_list):
        mask = crop_arr == crop
        vals = raw[mask]
        p5, p95 = np.percentile(vals, 5), np.percentile(vals, 95)
        norm[mask] = np.clip((vals - p5) / (p95 - p5 + 1e-9), 0, 1)
    return norm


stest_emb   = dino_emb[stest_meta["pos_idx"].values]
stest_cls   = np.array(stest_meta["class_name"].tolist())
stest_crops = [c.split("___")[0] for c in stest_cls]

utest_meta_eval = utest_meta.iloc[np.where(utest_mask)[0]].copy()
utest_emb   = dino_emb[utest_meta_eval["pos_idx"].values]
utest_cls   = np.array(utest_meta_eval["class_name"].tolist())
utest_crops = [c.split("___")[0] for c in utest_cls]

sev_seen   = compute_severity(stest_emb, stest_crops)
sev_unseen = compute_severity(utest_emb, utest_crops)

class_sev = {}
for cls in seen_classes:
    mask = stest_cls == cls
    if mask.sum():
        class_sev[cls] = sev_seen[mask]
for cls in unseen_classes:
    mask = utest_cls == cls
    if mask.sum():
        class_sev[cls] = sev_unseen[mask]

healthy_cls  = [c for c in seen_classes + unseen_classes if "Healthy" in c and c in class_sev]
diseased_cls = [c for c in seen_classes + unseen_classes if "Healthy" not in c and c in class_sev]
h_vals = np.concatenate([class_sev[c] for c in healthy_cls])
d_vals = np.concatenate([class_sev[c] for c in diseased_cls])
ratio  = d_vals.mean() / (h_vals.mean() + 1e-9)

valid = 0
tot   = 0
for crop in set(c.split("___")[0] for c in seen_classes + unseen_classes):
    hs = [class_sev[c].mean() for c in healthy_cls  if c.split("___")[0] == crop and c in class_sev]
    ds = [class_sev[c].mean() for c in diseased_cls if c.split("___")[0] == crop and c in class_sev]
    if hs and ds:
        tot += 1
        valid += int(np.mean(hs) < np.mean(ds))
grade_acc = valid / max(tot, 1)

print(f"  Healthy mean severity  : {h_vals.mean():.4f}")
print(f"  Diseased mean severity : {d_vals.mean():.4f}")
print(f"  Separation ratio       : {ratio:.2f}x")
print(f"  Per-crop grading acc   : {grade_acc*100:.1f}% ({valid}/{tot} crops)")


section(f"[13] Federated Learning ({FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs)")

client_data = {}
for client, crops in CLIENTS.items():
    ccls = [c for c in seen_classes if c.split("___")[0] in crops]
    ctr  = train_meta[train_meta["class_name"].isin(ccls)].copy()
    cval = val_meta[val_meta["class_name"].isin(ccls)].copy()
    client_data[client] = {"classes": ccls, "train": ctr, "val": cval, "n": len(ctr)}
    print(f"  {client:<12} crops={crops}  classes={len(ccls)}  n_train={len(ctr)}")


def get_weights(m):
    return {k: v.clone().detach() for k, v in m.state_dict().items()}


def set_weights(m, w):
    m.load_state_dict(w)


def fedavg_agg(weights_list, sizes):
    total = sum(sizes)
    avg   = {}
    for key in weights_list[0]:
        avg[key] = sum(w[key] * (n / total) for w, n in zip(weights_list, sizes))
    return avg


def local_train_fn(global_w, client_info, local_epochs, lr, model_class):
    lm  = model_class().to(DEVICE)
    lm.load_state_dict(global_w)
    lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=1e-3)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.05)
    df  = client_info["train"]
    if len(df) == 0:
        return get_weights(lm), 0, 0.
    idx    = df["pos_idx"].values
    lab    = torch.tensor(df["seen_id"].values, dtype=torch.long)
    losses = []
    for _ in range(local_epochs):
        perm = np.random.permutation(len(idx))
        for i in range(0, len(perm), LOCAL_BATCH):
            bx   = idx[perm[i:i + LOCAL_BATCH]]
            bl   = lab[perm[i:i + LOCAL_BATCH]]
            v    = dino_emb[bx].to(DEVICE)
            l    = bl.to(DEVICE)
            loss = ce(lm(v), l)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    lm.eval()
    return get_weights(lm), len(df), float(np.mean(losses)) if losses else 0.


def run_fl(model_class, label, alpha_use, gamma_use, T_use):
    global_m = model_class().to(DEVICE)
    metrics  = []
    best_H_val = 0.
    best_w   = None
    for rnd in range(1, FL_ROUNDS + 1):
        gw  = get_weights(global_m)
        cws = []
        csz = []
        for cname, cinfo in client_data.items():
            lw, n, _ = local_train_fn(gw, cinfo, LOCAL_EPOCHS, LOCAL_LR, model_class)
            if n > 0:
                cws.append(lw)
                csz.append(n)
        if cws:
            set_weights(global_m, fedavg_agg(cws, csz))
        global_m.eval()

        sm_v  = get_probe_sm(val_meta["pos_idx"].values, global_m)
        pfv   = expand_full(sm_v, len(v_labels), C, seen_ids_np)
        cv    = softmax_np(clip_v_raw,     T_use)
        cuv   = softmax_np(clip_u_raw_val, T_use)
        fv_fl = alpha_use * pfv + (1 - alpha_use) * cv
        fv_fl[:, seen_ids_np] -= gamma_use
        fuv_fl = (1 - alpha_use) * cuv
        fuv_fl[:, seen_ids_np] -= gamma_use
        _, _, S_val, U_val, H_val = gzsl_eval(
            fv_fl, fuv_fl, v_labels_np, u_labels_val, all_cls_arr)

        sm_s  = get_probe_sm(stest_meta["pos_idx"].values, global_m)
        pfs   = expand_full(sm_s, len(s_labels), C, seen_ids_np)
        cs    = softmax_np(clip_s_raw,      T_use)
        cu    = softmax_np(clip_u_raw_test, T_use)
        fs_fl = alpha_use * pfs + (1 - alpha_use) * cs
        fs_fl[:, seen_ids_np] -= gamma_use
        fu_fl = (1 - alpha_use) * cu
        fu_fl[:, seen_ids_np] -= gamma_use
        _, _, S_fl, U_fl, H_fl = gzsl_eval(
            fs_fl, fu_fl, s_labels_np, u_labels_test, all_cls_arr)

        zsl_fl = pacc(
            unseen_arr[(clip_u_raw_all[:, unseen_ids_np] / TEMP_ZSL).argmax(1)],
            u_lab_all_np)
        metrics.append({"round": rnd, "S": S_fl, "U": U_fl, "H": H_fl,
                         "S_val": S_val, "U_val": U_val, "H_val": H_val, "ZSL": zsl_fl})
        if rnd % 10 == 0 or rnd == 1:
            print(f"  [{label}] Round {rnd:3d}: S={S_fl*100:.2f}%  U={U_fl*100:.2f}%  "
                  f"H={H_fl*100:.2f}%  (val H={H_val*100:.2f}%)  ({elapsed()})")
        if H_val > best_H_val:
            best_H_val = H_val
            best_w = copy.deepcopy(get_weights(global_m))
    return metrics, best_H_val, best_w, global_m


random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

fl_metrics_p, fl_bestH_p, fl_bestw_p, fl_model_p = run_fl(
    LinearProbe, "BCVSA", af, gf, Tf)

set_weights(fl_model_p, fl_bestw_p)
fl_model_p.eval()
sm_s_fl = get_probe_sm(stest_meta["pos_idx"].values, fl_model_p)
pfs_fl  = expand_full(sm_s_fl, len(s_labels), C, seen_ids_np)

cs_fl   = softmax_np(clip_s_raw, Tf)
cu_fl   = softmax_np(clip_u_raw_test, Tf)
fs_fl_f = af * pfs_fl + (1 - af) * cs_fl
fs_fl_f[:, seen_ids_np] -= gf
fu_fl_f = (1 - af) * cu_fl
fu_fl_f[:, seen_ids_np] -= gf
pred_s_flp, pred_u_flp, S_flp, U_flp, H_flp = gzsl_eval(
    fs_fl_f, fu_fl_f, s_labels_np, u_labels_test, all_cls_arr)

probe_full_v_fl = expand_full(get_probe_sm(val_meta["pos_idx"].values, fl_model_p),
                               len(v_labels), C, seen_ids_np)
fl_cal = run_calibration_search(probe_full_v_fl, desc="  FL recalibration")
af_fl, gf_fl, Tf_fl = fl_cal["alpha"], fl_cal["gamma"], fl_cal["T_clip"]

cs_fl2  = softmax_np(clip_s_raw,      Tf_fl)
cu_fl2  = softmax_np(clip_u_raw_test, Tf_fl)
fs_fl2  = af_fl * pfs_fl + (1 - af_fl) * cs_fl2
fs_fl2[:, seen_ids_np] -= gf_fl
fu_fl2  = (1 - af_fl) * cu_fl2
fu_fl2[:, seen_ids_np] -= gf_fl
pred_s_flp2, pred_u_flp2, S_flp2, U_flp2, H_flp2 = gzsl_eval(
    fs_fl2, fu_fl2, s_labels_np, u_labels_test, all_cls_arr)
met_flp2_s = full_metrics(s_labels_np,   pred_s_flp2, seen_classes)
met_flp2_u = full_metrics(u_labels_test, pred_u_flp2, unseen_classes)

print(f"\n  FL-BCVSA (centralized calibration reused, for reference): "
      f"S={S_flp*100:.2f}%  U={U_flp*100:.2f}%  H={H_flp*100:.2f}%")
print(f"  FL-BCVSA (own calibration alpha={af_fl} gamma={gf_fl:.4f} T={Tf_fl} "
      f"-- report this one): S={S_flp2*100:.2f}%  U={U_flp2*100:.2f}%  H={H_flp2*100:.2f}%")
print(f"  Privacy cost (own calibration): {H_p*100 - H_flp2*100:.2f}pp")
print(f"  Checkpoint selected by validation H={fl_bestH_p*100:.2f}%; "
      f"S/U/H above are a single held-out test evaluation of that checkpoint.")


section("[13b] Local-Only Baselines (no federation) — per client")

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


def train_local_only(client_info, epochs, lr):
    lm  = LinearProbe().to(DEVICE)
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=1e-3)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.05)
    df  = client_info["train"]
    if len(df) == 0:
        return lm
    idx = df["pos_idx"].values
    lab = torch.tensor(df["seen_id"].values, dtype=torch.long)
    for _ in range(epochs):
        perm = np.random.permutation(len(idx))
        for i in range(0, len(perm), LOCAL_BATCH):
            bx = idx[perm[i:i + LOCAL_BATCH]]
            bl = lab[perm[i:i + LOCAL_BATCH]]
            v  = dino_emb[bx].to(DEVICE)
            l  = bl.to(DEVICE)
            opt.zero_grad()
            ce(lm(v), l).backward()
            opt.step()
    lm.eval()
    return lm


local_only_results = {}
for cname, cinfo in client_data.items():
    if cinfo["n"] == 0:
        print(f"  {cname}: no training data, skipping")
        continue
    local_model = train_local_only(cinfo, EPOCHS_PROBE, LR_PROBE)
    probe_full_v_local = expand_full(get_probe_sm(val_meta["pos_idx"].values, local_model),
                                      len(v_labels), C, seen_ids_np)
    local_cal = run_calibration_search(probe_full_v_local, desc=f"  {cname} calib")
    a_l, g_l, T_l = local_cal["alpha"], local_cal["gamma"], local_cal["T_clip"]

    sm_s_local = get_probe_sm(stest_meta["pos_idx"].values, local_model)
    pfs_local  = expand_full(sm_s_local, len(s_labels), C, seen_ids_np)
    cs_local   = softmax_np(clip_s_raw,      T_l)
    cu_local   = softmax_np(clip_u_raw_test, T_l)
    fs_local   = a_l * pfs_local + (1 - a_l) * cs_local
    fs_local[:, seen_ids_np] -= g_l
    fu_local   = (1 - a_l) * cu_local
    fu_local[:, seen_ids_np] -= g_l
    _, _, S_local, U_local, H_local = gzsl_eval(
        fs_local, fu_local, s_labels_np, u_labels_test, all_cls_arr)

    local_only_results[cname] = {
        "n_train": int(cinfo["n"]), "n_classes": len(cinfo["classes"]),
        "alpha": a_l, "gamma": round(g_l, 4), "T_clip": T_l,
        "S": round(S_local * 100, 2), "U": round(U_local * 100, 2),
        "H": round(H_local * 100, 2),
    }
    print(f"  {cname:<12} (local-only, n_train={cinfo['n']:>5}, "
          f"{len(cinfo['classes'])} classes): "
          f"S={S_local*100:.2f}%  U={U_local*100:.2f}%  H={H_local*100:.2f}%")

print(f"\n  Federated (FL-BCVSA, self-calibrated): "
      f"S={S_flp2*100:.2f}%  U={U_flp2*100:.2f}%  H={H_flp2*100:.2f}%")
print("  -> compare each client's local-only H above against this line:")
print("     any client whose local-only H is lower is a client federation "
      "is actually helping.")


section("[13c] Federated Learning with Differential Privacy (DP-FedAvg)")

if RUN_DP_FL:

    def clip_update(global_w, local_w, clip_norm):
        delta = {k: (local_w[k] - global_w[k]) for k in global_w}
        flat_norm = torch.sqrt(sum((v.float() ** 2).sum() for v in delta.values()))
        scale = min(1.0, clip_norm / (float(flat_norm) + 1e-12))
        return {k: v * scale for k, v in delta.items()}

    def fedavg_agg_dp(global_w, weights_list, clip_norm, noise_multiplier, verbose=False):
        L = len(weights_list)
        clipped = [clip_update(global_w, lw, clip_norm) for lw in weights_list]
        new_w = {}
        signal_sq = 0.0
        noise_sq  = 0.0
        for k in global_w:
            stacked    = torch.stack([c[k].float() for c in clipped], dim=0)
            mean_delta = stacked.mean(dim=0)
            noise_std  = noise_multiplier * clip_norm / L
            noise      = torch.randn_like(mean_delta) * noise_std
            if verbose:
                signal_sq += float((mean_delta ** 2).sum())
                noise_sq  += float((noise ** 2).sum())
            new_w[k]   = (global_w[k].float() + mean_delta + noise).to(global_w[k].dtype)
        if verbose:
            signal_norm = signal_sq ** 0.5
            noise_norm  = noise_sq ** 0.5
            ratio = noise_norm / (signal_norm + 1e-12)
            warn = "  <-- WARNING: noise dominates signal, lower DP_NOISE_MULTIPLIER" if ratio > 2 else ""
            print(f"    [DP diagnostic] ||signal||={signal_norm:.4f}  ||noise||={noise_norm:.4f}  "
                  f"noise/signal={ratio:.2f}{warn}")
        return new_w

    def run_fl_dp(model_class, label, alpha_use, gamma_use, T_use,
                  clip_norm, noise_multiplier):
        global_m = model_class().to(DEVICE)
        metrics  = []
        best_H_val = 0.
        best_w   = None
        for rnd in range(1, FL_ROUNDS + 1):
            gw  = get_weights(global_m)
            cws = []
            for cname, cinfo in client_data.items():
                lw, n, _ = local_train_fn(gw, cinfo, LOCAL_EPOCHS, LOCAL_LR, model_class)
                if n > 0:
                    cws.append(lw)
            if cws:
                set_weights(global_m, fedavg_agg_dp(
                    gw, cws, clip_norm, noise_multiplier,
                    verbose=(rnd == 1 or rnd % 25 == 0)))
            global_m.eval()

            sm_v  = get_probe_sm(val_meta["pos_idx"].values, global_m)
            pfv   = expand_full(sm_v, len(v_labels), C, seen_ids_np)
            cv    = softmax_np(clip_v_raw,     T_use)
            cuv   = softmax_np(clip_u_raw_val, T_use)
            fv_fl = alpha_use * pfv + (1 - alpha_use) * cv
            fv_fl[:, seen_ids_np] -= gamma_use
            fuv_fl = (1 - alpha_use) * cuv
            fuv_fl[:, seen_ids_np] -= gamma_use
            _, _, S_val, U_val, H_val = gzsl_eval(
                fv_fl, fuv_fl, v_labels_np, u_labels_val, all_cls_arr)

            sm_s  = get_probe_sm(stest_meta["pos_idx"].values, global_m)
            pfs   = expand_full(sm_s, len(s_labels), C, seen_ids_np)
            cs    = softmax_np(clip_s_raw,      T_use)
            cu    = softmax_np(clip_u_raw_test, T_use)
            fs_fl = alpha_use * pfs + (1 - alpha_use) * cs
            fs_fl[:, seen_ids_np] -= gamma_use
            fu_fl = (1 - alpha_use) * cu
            fu_fl[:, seen_ids_np] -= gamma_use
            _, _, S_fl, U_fl, H_fl = gzsl_eval(
                fs_fl, fu_fl, s_labels_np, u_labels_test, all_cls_arr)

            metrics.append({"round": rnd, "S": S_fl, "U": U_fl, "H": H_fl,
                             "S_val": S_val, "U_val": U_val, "H_val": H_val})
            if rnd % 10 == 0 or rnd == 1:
                print(f"  [{label}] Round {rnd:3d}: S={S_fl*100:.2f}%  U={U_fl*100:.2f}%  "
                      f"H={H_fl*100:.2f}%  (val H={H_val*100:.2f}%)  ({elapsed()})")
            if H_val > best_H_val:
                best_H_val = H_val
                best_w = copy.deepcopy(get_weights(global_m))
        return metrics, best_H_val, best_w, global_m

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    fl_metrics_dp, fl_bestH_dp, fl_bestw_dp, fl_model_dp = run_fl_dp(
        LinearProbe, "BCVSA-DP", af, gf, Tf, DP_CLIP_NORM, DP_NOISE_MULTIPLIER)

    set_weights(fl_model_dp, fl_bestw_dp)
    fl_model_dp.eval()
    sm_s_dp = get_probe_sm(stest_meta["pos_idx"].values, fl_model_dp)
    pfs_dp  = expand_full(sm_s_dp, len(s_labels), C, seen_ids_np)
    probe_full_v_dp = expand_full(get_probe_sm(val_meta["pos_idx"].values, fl_model_dp),
                                   len(v_labels), C, seen_ids_np)
    dp_cal = run_calibration_search(probe_full_v_dp, desc="  DP-FL recalibration")
    af_dp, gf_dp, Tf_dp = dp_cal["alpha"], dp_cal["gamma"], dp_cal["T_clip"]

    cs_dp = softmax_np(clip_s_raw,      Tf_dp)
    cu_dp = softmax_np(clip_u_raw_test, Tf_dp)
    fs_dp = af_dp * pfs_dp + (1 - af_dp) * cs_dp
    fs_dp[:, seen_ids_np] -= gf_dp
    fu_dp = (1 - af_dp) * cu_dp
    fu_dp[:, seen_ids_np] -= gf_dp
    pred_s_dp, pred_u_dp, S_dp, U_dp, H_dp = gzsl_eval(
        fs_dp, fu_dp, s_labels_np, u_labels_test, all_cls_arr)
    met_dp_s = full_metrics(s_labels_np,   pred_s_dp, seen_classes)
    met_dp_u = full_metrics(u_labels_test, pred_u_dp, unseen_classes)

    print(f"\n  FL-BCVSA + DP (clip={DP_CLIP_NORM}, noise_mult={DP_NOISE_MULTIPLIER}): "
          f"S={S_dp*100:.2f}%  U={U_dp*100:.2f}%  H={H_dp*100:.2f}%")
    print(f"  Utility cost of DP on top of federation: {H_flp2*100 - H_dp*100:.2f}pp "
          f"(federation alone already cost {H_p*100 - H_flp2*100:.2f}pp vs. centralized)")
    print(f"  Checkpoint selected by validation H={fl_bestH_dp*100:.2f}%; "
          f"S/U/H above are a single held-out test evaluation of that checkpoint.")
    print("  Mechanism only (McMahan et al. 2018): no (epsilon, delta) accountant "
          "has been run for this (clip_norm, noise_multiplier, rounds, client_count) "
          "configuration -- do not report a specific epsilon without one.")
else:
    fl_metrics_dp = []
    af_dp = gf_dp = Tf_dp = None
    H_dp = S_dp = U_dp = None
    met_dp_s = met_dp_u = {"f1": None}


section("[14] Saving Results")

master_rows = [
    {
        "Method":       "Ablation A: CLIP Only",
        "ZSL%":         round(best_zsl_a * 100, 2),
        "S%":           round(acc_s_a * 100, 2),
        "U%":           round(acc_u_a * 100, 2),
        "H%":           round(H_a * 100, 2),
        "H_3shot%":     "-",
        "H_5shot%":     "-",
        "F1_Seen%":     round(met_a_s["f1"] * 100, 2),
        "F1_Unseen%":   round(met_a_u["f1"] * 100, 2),
        "ECE_Seen%":    round(ece_a_s * 100, 2),
        "ECE_Unseen%":  round(ece_a_u * 100, 2),
        "T_clip_or_temp": best_a["T"],
        "gamma":          round(best_a["gamma"], 4),
        "Val_H_during_tuning%": round(best_a["H"] * 100, 2),
    },
    {
        "Method":       "Ablation B: DINOv2 Probe (Supervised)",
        "ZSL%":         "N/A",
        "S%":           round(met_b["acc"] * 100, 2),
        "U%":           "N/A",
        "H%":           "N/A",
        "H_3shot%":     "-",
        "H_5shot%":     "-",
        "F1_Seen%":     round(met_b["f1"] * 100, 2),
        "F1_Unseen%":   "N/A",
        "ECE_Seen%":    "N/A",
        "ECE_Unseen%":  "N/A",
        "T_clip_or_temp": "N/A",
        "gamma":          "N/A",
        "Val_H_during_tuning%": "N/A",
    },
    {
        "Method":       "Ablation C: Alignment Head",
        "ZSL%":         round(best_zsl_c * 100, 2),
        "S%":           round(acc_s_c * 100, 2),
        "U%":           round(acc_u_c * 100, 2),
        "H%":           round(H_c * 100, 2),
        "H_3shot%":     round(H3_c * 100, 2),
        "H_5shot%":     round(H5_c * 100, 2),
        "F1_Seen%":     round(met_c_s["f1"] * 100, 2),
        "F1_Unseen%":   round(met_c_u["f1"] * 100, 2),
        "ECE_Seen%":    round(ece_c_s * 100, 2),
        "ECE_Unseen%":  round(ece_c_u * 100, 2),
        "T_clip_or_temp": tc,
        "gamma":          round(gc, 4),
        "Val_H_during_tuning%": round(best_c["H"] * 100, 2),
    },
    {
        "Method":       "BCVSA (Proposed) 0-shot",
        "ZSL%":         round(acc_zsl_p * 100, 2),
        "S%":           round(acc_s_p * 100, 2),
        "U%":           round(acc_u_p * 100, 2),
        "H%":           round(H_p * 100, 2),
        "H_3shot%":     round(H3_p * 100, 2),
        "H_5shot%":     round(H5_p * 100, 2),
        "F1_Seen%":     round(met_p_s["f1"] * 100, 2),
        "F1_Unseen%":   round(met_p_u["f1"] * 100, 2),
        "ECE_Seen%":    round(ece_p_s * 100, 2),
        "ECE_Unseen%":  round(ece_p_u * 100, 2),
        "T_clip_or_temp": Tf,
        "gamma":          round(gf, 4),
        "Val_H_during_tuning%": round(best_cal["H"] * 100, 2),
    },
    {
        "Method":       "FL-BCVSA (Federated, 4 Clients, self-calibrated)",
        "ZSL%":         round(pacc(
                            unseen_arr[(clip_u_raw_all[:, unseen_ids_np] / TEMP_ZSL).argmax(1)],
                            u_lab_all_np) * 100, 2),
        "S%":           round(S_flp2 * 100, 2),
        "U%":           round(U_flp2 * 100, 2),
        "H%":           round(H_flp2 * 100, 2),
        "H_3shot%":     "-",
        "H_5shot%":     "-",
        "F1_Seen%":     round(met_flp2_s["f1"] * 100, 2),
        "F1_Unseen%":   round(met_flp2_u["f1"] * 100, 2),
        "ECE_Seen%":    "-",
        "ECE_Unseen%":  "-",
        "T_clip_or_temp": Tf_fl,
        "gamma":          round(gf_fl, 4),
        "Val_H_during_tuning%": round(fl_cal["H"] * 100, 2),
    },
]
df_master = pd.DataFrame(master_rows)
df_master.to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)

for name, df in [
    ("error_analysis_unseen",    df_err_p_u),
    ("error_analysis_seen",      df_err_p_s),
    ("error_analysis_ablc_unseen", df_err_c_u),
    ("error_analysis_ablc_seen",   df_err_c_s),
    ("confusion_pairs_unseen",   df_wp_p_u),
    ("confusion_pairs_ablc",     df_wp_c_u),
]:
    df.to_csv(os.path.join(RESULTS_DIR, f"{name}.csv"), index=False)

pd.concat([df_hd_p_u, df_hd_p_s, df_hd_c_u, df_hd_c_s]).to_csv(
    os.path.join(RESULTS_DIR, "healthy_diseased_breakdown.csv"), index=False)

sev_rows = []
for cls in seen_classes + unseen_classes:
    if cls not in class_sev:
        continue
    sev_rows.append({
        "Class":      cls,
        "Type":       "seen" if cls in seen_classes else "unseen",
        "Is_Healthy": "Healthy" in cls,
        "Mean_Sev":   round(float(class_sev[cls].mean()), 4),
        "Std_Sev":    round(float(class_sev[cls].std()),  4),
    })
pd.DataFrame(sev_rows).to_csv(os.path.join(RESULTS_DIR, "severity_per_class.csv"), index=False)

pd.DataFrame(fl_metrics_p).to_csv(os.path.join(RESULTS_DIR, "fl_convergence_bcvsa.csv"), index=False)

if RUN_MULTI_SEED and len(multi_seed_H3) > 1:
    few_shot_str = (f"mean+/-std ({len(multi_seed_H3)} seeds): "
                    f"3-shot H={np.mean(multi_seed_H3)*100:.2f}+/-{np.std(multi_seed_H3)*100:.2f}  "
                    f"5-shot H={np.mean(multi_seed_H5)*100:.2f}+/-{np.std(multi_seed_H5)*100:.2f}")
else:
    few_shot_str = f"3-shot H={H3_p*100:.2f}%  5-shot H={H5_p*100:.2f}%  (seed=42)"

summary = {
    "protocol":    "labeled_validation_split_50_50_seed42",
    "calibration": {"alpha": af, "gamma": round(gf, 4), "T_clip": Tf},
    "proposed_0shot": {
        "ZSL":  round(acc_zsl_p * 100, 2), "S": round(acc_s_p * 100, 2),
        "U":    round(acc_u_p * 100, 2),   "H": round(H_p * 100, 2),
        "F1_seen":   round(met_p_s["f1"] * 100, 2),
        "F1_unseen": round(met_p_u["f1"] * 100, 2),
        "ECE_seen":  round(ece_p_s * 100, 2),
        "ECE_unseen":round(ece_p_u * 100, 2),
        "ZSL_N":     len(u_lab_all),
        "GZSL_N_test": N_utest,
    },
    "proposed_3shot": {
        "ZSL": round(zsl3_p * 100, 2), "S": round(s3_p * 100, 2),
        "U":   round(u3_p * 100, 2),   "H": round(H3_p * 100, 2),
        "H_mean": round(float(np.mean(multi_seed_H3)) * 100, 2) if RUN_MULTI_SEED else None,
        "H_std":  round(float(np.std(multi_seed_H3)) * 100, 2) if RUN_MULTI_SEED else None,
        "n_seeds": len(multi_seed_H3) if RUN_MULTI_SEED else 1,
    },
    "proposed_5shot": {
        "ZSL": round(zsl5_p * 100, 2), "S": round(s5_p * 100, 2),
        "U":   round(u5_p * 100, 2),   "H": round(H5_p * 100, 2),
        "H_mean": round(float(np.mean(multi_seed_H5)) * 100, 2) if RUN_MULTI_SEED else None,
        "H_std":  round(float(np.std(multi_seed_H5)) * 100, 2) if RUN_MULTI_SEED else None,
        "n_seeds": len(multi_seed_H5) if RUN_MULTI_SEED else 1,
    },
    "few_shot_summary": few_shot_str,
    "ablation_a": {
        "ZSL": round(best_zsl_a * 100, 2), "S": round(acc_s_a * 100, 2),
        "U":   round(acc_u_a * 100, 2),    "H": round(H_a * 100, 2),
        "ECE_seen": round(ece_a_s * 100, 2), "ECE_unseen": round(ece_a_u * 100, 2),
    },
    "ablation_b": {
        "supervised_acc": round(met_b["acc"] * 100, 2),
        "F1": round(met_b["f1"] * 100, 2),
    },
    "ablation_c_0shot": {
        "ZSL": round(best_zsl_c * 100, 2), "S": round(acc_s_c * 100, 2),
        "U":   round(acc_u_c * 100, 2),    "H": round(H_c * 100, 2),
        "ECE_seen": round(ece_c_s * 100, 2), "ECE_unseen": round(ece_c_u * 100, 2),
    },
    "ablation_c_3shot": {
        "ZSL": round(zsl3_c * 100, 2), "H": round(H3_c * 100, 2),
        "H_mean": round(float(np.mean(multi_seed_H3_c)) * 100, 2) if RUN_MULTI_SEED else None,
        "H_std":  round(float(np.std(multi_seed_H3_c)) * 100, 2) if RUN_MULTI_SEED else None,
    },
    "ablation_c_5shot": {
        "ZSL": round(zsl5_c * 100, 2), "H": round(H5_c * 100, 2),
        "H_mean": round(float(np.mean(multi_seed_H5_c)) * 100, 2) if RUN_MULTI_SEED else None,
        "H_std":  round(float(np.std(multi_seed_H5_c)) * 100, 2) if RUN_MULTI_SEED else None,
    },
    "fl_bcvsa": {
        "alpha": af_fl, "gamma": round(gf_fl, 4), "T_clip": Tf_fl,
        "H":           round(H_flp2 * 100, 2),
        "S":           round(S_flp2 * 100, 2),
        "U":           round(U_flp2 * 100, 2),
        "privacy_cost_pp": round(H_p * 100 - H_flp2 * 100, 2),
        "H_reused_centralized_calibration": round(H_flp * 100, 2),
        "checkpoint_selected_by": "validation_H",
        "U_constant":  True,
    },
    "local_only_baselines": local_only_results,
    "fl_bcvsa_dp": {
        "enabled": RUN_DP_FL,
        "mechanism": "DP-FedAvg (McMahan et al. 2018): per-client update clipping + "
                     "Gaussian noise on the aggregate",
        "clip_norm": DP_CLIP_NORM if RUN_DP_FL else None,
        "noise_multiplier": DP_NOISE_MULTIPLIER if RUN_DP_FL else None,
        "checkpoint_selected_by": "validation_H",
        "alpha": af_dp, "gamma": round(gf_dp, 4) if gf_dp is not None else None, "T_clip": Tf_dp,
        "H": round(H_dp * 100, 2) if H_dp is not None else None,
        "S": round(S_dp * 100, 2) if S_dp is not None else None,
        "U": round(U_dp * 100, 2) if U_dp is not None else None,
        "F1_seen":   round(met_dp_s["f1"] * 100, 2) if met_dp_s["f1"] is not None else None,
        "F1_unseen": round(met_dp_u["f1"] * 100, 2) if met_dp_u["f1"] is not None else None,
        "utility_cost_vs_federated_pp": round(H_flp2 * 100 - H_dp * 100, 2) if H_dp is not None else None,
        "privacy_accounting_note": "Mechanism only. No certified (epsilon, delta) budget "
                                    "has been computed -- do not quote a specific epsilon "
                                    "in the paper without running a proper accountant.",
    },
    "severity": {
        "healthy_mean":  round(float(h_vals.mean()), 4),
        "diseased_mean": round(float(d_vals.mean()), 4),
        "ratio":         round(float(ratio), 3),
        "grading_acc_%": round(grade_acc * 100, 1),
        "n_classes":     len([c for c in unseen_classes if c in class_sev]),
    },
    "methodology_notes": {
        "ece_seen": "Computed in logit space (probe raw logits + CLIP similarity/Tf) "
                    "for the proposed method and both ablations, avoiding "
                    "probability-clipping distortion.",
        "severity": "Crop-specific healthy/diseased centroids for crops absent from the "
                    "seen set are built from the unseen CALIBRATION split only; "
                    "separation/grading metrics are reported on the held-out EVALUATION "
                    "split only.",
        "fl_bcvsa": "Reported H uses calibration parameters refit on the federated "
                    "model's own outputs. FL/DP-FL checkpoints are selected by "
                    "validation H each round; S/U/H reported is a single held-out "
                    "test evaluation of that selected checkpoint.",
        "t_clip_search": "T_clip is searched in DESCENDING order with a strict-"
                    "improvement tie-break, keeping the largest T that already "
                    "reaches the ceiling H rather than an arbitrary minimal value.",
        "ablation_c_zsl": "T/alpha for Ablation C's ZSL number are selected on the "
                    "unseen CALIBRATION half only and the resulting accuracy is "
                    "reported on the unseen EVALUATION half only.",
    },
}
with open(os.path.join(RESULTS_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

tick("All results saved")
print(df_master.to_string(index=False))


section("[15] Generating Figures")

import matplotlib.patches as mpatches2

fig1, ax1 = plt.subplots(figsize=(16, 9))
ax1.set_xlim(0, 16)
ax1.set_ylim(0, 9)
ax1.axis("off")


def draw_box(ax, x, y, w, h, label, sublabel="", color="#2563EB", tc="white", fs=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                edgecolor="black", linewidth=1.5, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=4)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                ha="center", va="center", fontsize=fs - 1.5, color=tc, zorder=4)


def arr(ax, x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)


draw_box(ax1, 0.3, 4.0, 1.6, 1.0, "Input Image $I$", "H×W×3", color="#374151")
arr(ax1, 1.9, 4.8, 2.8, 6.4)
arr(ax1, 1.9, 4.2, 2.8, 3.0)
draw_box(ax1, 2.8, 5.9, 2.8, 1.0, "DINOv2-Large", "(frozen)", color="#2563EB")
arr(ax1, 5.6, 6.4, 6.4, 6.4)
arr(ax1, 5.6, 3.0, 6.4, 3.0)
draw_box(ax1, 6.4, 5.9, 2.5, 1.0, "Linear Probe $f_\\theta$",
         f"1024→512→256→{S_n}", color="#0D9488")
ax1.text(7.65, 7.1, "[trainable]", fontsize=8, color="#0D9488")
arr(ax1, 8.9, 6.4, 9.7, 6.4)
draw_box(ax1, 2.8, 2.5, 2.8, 1.0, "CLIP ViT-L/14", "(frozen)", color="#EA580C")
draw_box(ax1, 0.3, 1.0, 2.0, 0.9, "10 Prompts/class", "crop + condition", color="#7C3AED", fs=8)
draw_box(ax1, 2.8, 1.0, 2.8, 0.9, "CLIP Text Encoder", "(frozen)", color="#EA580C", fs=8.5)
arr(ax1, 2.3, 1.45, 2.8, 1.45)
arr(ax1, 5.6, 1.45, 6.4, 1.45)
draw_box(ax1, 6.4, 2.5, 2.5, 1.0, "Cosine Similarity",
         r"$\cos(\hat{v},\bar{t}_c)/T$", color="#EA580C")
arr(ax1, 6.9, 1.9, 6.9, 2.5)
arr(ax1, 8.9, 3.0, 9.7, 3.0)
ax1.add_patch(plt.Rectangle((9.7, 3.8), 3.5, 2.8, facecolor="#F3F4F6",
                              edgecolor="black", linewidth=1.2, zorder=3))
ax1.text(10.05, 6.3,  "Bias Calibration Module", fontsize=10, fontweight="bold", color="#1F2937")
ax1.text(10.05, 5.8,
         r"$s_c = \alpha\,\tilde{s}^{\mathrm{probe}}_c + (1{-}\alpha)\,\tilde{s}^{\mathrm{CLIP}}_c$",
         fontsize=9.5, color="#1F2937")
ax1.text(10.05, 5.35,
         r"$\qquad - \gamma\,\mathbf{1}[c \in \mathcal{C}_s]$",
         fontsize=9.5, color="#1F2937")
for bx, by, txt, bc in [
    (10.2, 4.8, f"α={af}",       "#16A34A"),
    (11.2, 4.8, f"γ={gf:.3f}",   "#DC2626"),
    (12.1, 4.8, f"T={Tf}",       "#2563EB"),
]:
    ax1.add_patch(plt.Rectangle((bx, by), 0.85, 0.38, facecolor=bc,
                                 edgecolor="none", zorder=5, alpha=0.85))
    ax1.text(bx + 0.425, by + 0.19, txt, ha="center", va="center",
             fontsize=8, fontweight="bold", color="white", zorder=6)
ax1.text(10.05, 4.55, "Tuned on validation set only",
         fontsize=7.5, color="#6B7280", style="italic")
arr(ax1, 9.7, 6.4, 10.0, 6.2, color="#2563EB")
arr(ax1, 9.7, 3.0, 10.0, 4.0, color="#EA580C")
draw_box(ax1, 13.4, 4.3, 2.4, 1.4, "GZSL Prediction $\\hat{c}$",
         f"argmax over C={C} classes", color="#16A34A", fs=8.5)
arr(ax1, 13.2, 5.2, 13.4, 5.0, color="#6B7280")
for by2, txt2, bc2 in [
    (7.5, f"0-shot H = {H_p*100:.2f}%",  "#16A34A"),
    (7.1, f"5-shot H = {H5_p*100:.2f}%", "#2563EB"),
]:
    ax1.add_patch(plt.Rectangle((13.3, by2), 2.5, 0.35, facecolor=bc2,
                                 edgecolor="none", zorder=5, alpha=0.85))
    ax1.text(14.55, by2 + 0.175, txt2, ha="center", va="center",
             fontsize=8.5, fontweight="bold", color="white", zorder=6)
draw_box(ax1, 6.4, 0.05, 3.5, 0.85, "Bipolar Severity Estimator",
         r"$\sigma(I)=\cos(d,e_r)-\cos(d,h_r)\;\rightarrow\;\sigma\in[0,1]$",
         color="#D97706", fs=8)
ax1.annotate("", xy=(7.0, 0.9), xytext=(7.0, 5.9),
             arrowprops=dict(arrowstyle="-|>", color="#D97706", lw=1.2,
                             linestyle="dashed"), zorder=2)
ax1.legend(handles=[
    mpatches2.Patch(color="#2563EB", label="DINOv2 components"),
    mpatches2.Patch(color="#EA580C", label="CLIP components"),
    mpatches2.Patch(color="#0D9488", label="Trainable (linear probe only)"),
    mpatches2.Patch(color="#7C3AED", label="Text prompts"),
    mpatches2.Patch(color="#D97706", label="Severity estimator"),
    mpatches2.Patch(color="#16A34A", label="Output"),
], loc="lower left", bbox_to_anchor=(0.0, -0.01), ncol=3, fontsize=8, frameon=True)
ax1.set_title(
    "BCVSA: Bias-Calibrated Visual-Semantic Alignment Framework\n"
    "(Only the linear probe is trained; all other components are frozen)",
    fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
savefig(fig1, "figure1_architecture.png")

methods_f2 = ["CLIP Only\n(Abl. A)", "DINOv2 Only\n(Abl. B)", "Alignment\nHead (Abl. C)",
              "BCVSA\n0-shot", "BCVSA\n3-shot", "BCVSA\n5-shot"]
S_f2 = [acc_s_a * 100, met_b["acc"] * 100, acc_s_c * 100, acc_s_p * 100, s3_p * 100, s5_p * 100]
U_f2 = [acc_u_a * 100, 0.,                acc_u_c * 100, acc_u_p * 100, u3_p * 100, u5_p * 100]
H_f2 = [H_a * 100,     0.,                H_c * 100,     H_p * 100,     H3_p * 100, H5_p * 100]
x2 = np.arange(len(methods_f2))
w2 = 0.25
fig2, ax2 = plt.subplots(figsize=(13, 5.5))
for vals, offset, lbl, col in [
    (S_f2, -w2, "Seen Accuracy (S%)",    "#2563EB"),
    (U_f2,  0,  "Unseen Accuracy (U%)",  "#DC2626"),
    (H_f2, +w2, "Harmonic Mean (H%)",    "#16A34A"),
]:
    bars = ax2.bar(x2 + offset, vals, w2, label=lbl, color=col, alpha=0.88, edgecolor="white")
    for rect, v in zip(bars, vals):
        if v > 2:
            ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.5,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
ax2.set_xticks(x2)
ax2.set_xticklabels(methods_f2, fontsize=9)
ax2.set_ylim(0, 108)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_title("Ablation Study: Seen (S%), Unseen (U%), and Harmonic Mean (H%)",
              fontweight="bold", fontsize=11)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.3, axis="y")
ax2.axvline(x=2.5, color="gray", lw=1, linestyle="--", alpha=0.5)
ax2.text(2.65, 102, "Ablations", fontsize=8, color="gray")
ax2.text(3.15, 102, "|  Proposed Method", fontsize=8, color="gray")
plt.tight_layout()
savefig(fig2, "figure2_ablation_comparison.png")

alphas_g    = sorted(set(round(a, 2) for a, g in alpha_gamma_grid))
gammas_g    = sorted(set(round(g, 4) for a, g in alpha_gamma_grid))
g_step      = max(1, len(gammas_g) // 40)
gammas_plot = gammas_g[::g_step]
H_grid = np.zeros((len(alphas_g), len(gammas_plot)))
for i, a in enumerate(alphas_g):
    for j, g in enumerate(gammas_plot):
        H_grid[i, j] = alpha_gamma_grid.get((round(a, 2), round(g, 4)), 0) * 100

fig3, ax3 = plt.subplots(figsize=(12, 5))
im = ax3.imshow(H_grid, aspect="auto", origin="lower", cmap="RdYlGn", vmin=40, vmax=H_grid.max())
plt.colorbar(im, ax=ax3, label="GZSL Harmonic Mean H% (validation set)", fraction=0.03, pad=0.02)
ax3.set_xlabel("Bias Penalty γ (seen-class logit penalty)", fontsize=11)
ax3.set_ylabel("Mixture Weight α (probe contribution)", fontsize=11)
ax3.set_yticks(range(len(alphas_g)))
ax3.set_yticklabels([f"{a:.1f}" for a in alphas_g], fontsize=8)
if af in alphas_g:
    ai = alphas_g.index(round(af, 2))
    gi = min(range(len(gammas_plot)), key=lambda j: abs(gammas_plot[j] - gf))
    ax3.plot(gi, ai, "r*", markersize=16, label=f"Optimal: α={af}, γ={gf:.3f}")
    ax3.legend(fontsize=9)
ax3.set_title(f"Calibration Landscape: H% vs α and γ  (T={Tf})",
              fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig3, "figure3_calibration_heatmap.png")


def plot_cm(yt, yp, labs, title, ax):
    cm  = confusion_matrix(yt, yp, labels=labs)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    short = [c.split("___")[-1].replace("_", " ")[:13] for c in labs]
    n  = len(labs)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6 if n > 20 else 7)
    ax.set_yticklabels(short, fontsize=6 if n > 20 else 7)
    for i in range(n):
        for j in range(n):
            if cmn[i, j] > 0.05:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        fontsize=4 if n > 20 else 5.5,
                        color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    return im


fig4, axes4 = plt.subplots(1, 2, figsize=(22, 9))
im1 = plot_cm(s_labels_np,   pred_s_p, seen_classes,
              "Seen Classes — GZSL 0-shot (Proposed)",   axes4[0])
im2 = plot_cm(u_labels_test, pred_u_p, unseen_classes,
              "Unseen Classes — GZSL 0-shot (Proposed)", axes4[1])
plt.colorbar(im1, ax=axes4[0], fraction=0.04, pad=0.04)
plt.colorbar(im2, ax=axes4[1], fraction=0.04, pad=0.04)
fig4.suptitle("Normalised Confusion Matrices — BCVSA 0-shot GZSL",
              fontsize=12, fontweight="bold")
plt.tight_layout()
savefig(fig4, "figure4_confusion_matrices.png")

fig5, axes5 = plt.subplots(1, 2, figsize=(18, 7))
green_p = mpatches.Patch(color="#2ecc71", alpha=0.8, label="Healthy")
red_p   = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Diseased")
for ax, cls_list, split_name in [
    (axes5[0], seen_classes,   "Seen Classes"),
    (axes5[1], unseen_classes, "Unseen Classes"),
]:
    plot_cls = [c for c in cls_list if c in class_sev]
    if not plot_cls:
        ax.set_title(f"{split_name} — no data")
        continue
    plot_cls_sorted = sorted(plot_cls,
                             key=lambda c: (c.split("___")[0], "Healthy" not in c))
    colors = ["#2ecc71" if "Healthy" in c else "#e74c3c" for c in plot_cls_sorted]
    data   = [class_sev[c] for c in plot_cls_sorted]
    labels = [c.replace("___", "  ").replace("_", " ")[:18] for c in plot_cls_sorted]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", ms=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("Severity Score (0 = Healthy, 1 = Severe)", fontsize=9)
    ax.set_title(f"{split_name} — DINOv2 Bipolar Severity", fontsize=10, fontweight="bold")
    ax.axhline(y=h_vals.mean(), color="green", ls="--", lw=1.5, label="Healthy mean")
    ax.legend(handles=[green_p, red_p], fontsize=8)
    ax.grid(alpha=0.3, axis="y")

fig5.suptitle(
    f"Severity Estimation — Grading Accuracy: {grade_acc*100:.0f}%  |  "
    f"Separation Ratio: {ratio:.2f}x",
    fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig5, "figure5_severity_boxplots.png")

fig6, axes6 = plt.subplots(1, 2, figsize=(15, 5))
rounds_p = [m["round"] for m in fl_metrics_p]
H_fl_p   = [m["H"] * 100 for m in fl_metrics_p]
S_fl_p   = [m["S"] * 100 for m in fl_metrics_p]
U_fl_p   = [m["U"] * 100 for m in fl_metrics_p]

axes6[0].plot(rounds_p, H_fl_p, "b-", lw=2,
              label=f"FL-BCVSA, reused calibration (best val H={fl_bestH_p*100:.2f}%)")
axes6[0].axhline(y=H_p * 100, color="blue", ls=":", lw=1.8,
                 label=f"Centralised BCVSA = {H_p*100:.2f}%", alpha=0.7)
axes6[0].axhline(y=H_flp2 * 100, color="darkorange", ls="-.", lw=1.8,
                 label=f"FL-BCVSA, self-calibrated (final) = {H_flp2*100:.2f}%")
axes6[0].set_xlabel("Federated Round")
axes6[0].set_ylabel("GZSL Harmonic Mean H (%)")
axes6[0].set_title("Federated Learning Convergence — H%", fontweight="bold")
axes6[0].legend(fontsize=8)
axes6[0].grid(alpha=0.3)
axes6[0].set_ylim(0, 100)

axes6[1].plot(rounds_p, S_fl_p, "b-", lw=1.8, label="FL-BCVSA Seen Accuracy S%")
axes6[1].plot(rounds_p, U_fl_p, "r-", lw=1.8,
              label="FL-BCVSA Unseen Accuracy U% (constant by design)")
axes6[1].axhline(y=S_flp2 * 100, color="blue", ls=":", alpha=0.6)
axes6[1].axhline(y=U_flp2 * 100, color="red",  ls=":", alpha=0.6)
axes6[1].set_xlabel("Federated Round")
axes6[1].set_ylabel("Accuracy (%)")
axes6[1].set_title("Federated Learning — Seen vs Unseen Accuracy",
                   fontweight="bold", fontsize=9)
axes6[1].legend(fontsize=8)
axes6[1].grid(alpha=0.3)
fig6.suptitle(
    f"Federated Learning ({FL_ROUNDS} rounds, {len(CLIENTS)} clients) | "
    f"Privacy cost (self-calibrated): {H_p*100 - H_flp2*100:.2f} pp",
    fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig6, "figure6_fl_convergence.png")

fig7, axes7 = plt.subplots(1, 2, figsize=(18, 8))
for ax, df, title in [
    (axes7[0], df_wp_p_u, "BCVSA (Proposed) — Unseen Classes"),
    (axes7[1], df_wp_c_u, "Ablation C (Alignment Head) — Unseen Classes"),
]:
    if len(df) == 0:
        ax.set_title(f"{title}\n(no errors)")
        continue
    top    = df.head(15)
    labels = [f"{r['True_s'][:22]} → {r['Pred_s'][:22]}" for _, r in top.iterrows()]
    ax.barh(range(len(top)), top["Count"].values, color="#e74c3c", alpha=0.8,
            edgecolor="k", lw=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i, v in enumerate(top["Count"].values):
        ax.text(v + 1, i, str(v), va="center", fontsize=7.5)
    ax.set_xlabel("Misclassification Count")
    ax.set_title(f"{title}\nTop-15 Confusion Pairs", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
fig7.suptitle("Misclassification Analysis — Unseen Classes",
              fontsize=12, fontweight="bold")
plt.tight_layout()
savefig(fig7, "figure7_misclassification_analysis.png")

tick("Computing t-SNE embedding...")
tsne_idx  = []
tsne_lab  = []
tsne_type = []
for cls in seen_classes:
    idx = stest_meta[stest_meta["class_name"] == cls]["pos_idx"].values
    sel = idx[:min(len(idx), 150)]
    tsne_idx.extend(sel)
    tsne_lab.extend([cls] * len(sel))
    tsne_type.extend(["seen"] * len(sel))
for cls in unseen_classes:
    idx = utest_meta[utest_meta["class_name"] == cls]["pos_idx"].values
    sel = idx[:min(len(idx), 150)]
    tsne_idx.extend(sel)
    tsne_lab.extend([cls] * len(sel))
    tsne_type.extend(["unseen"] * len(sel))

tsne_emb = dino_emb[np.array(tsne_idx)].numpy()
n_tsne   = len(tsne_emb)
perp     = min(40, max(5, n_tsne // 3 - 1))
print(f"  t-SNE: n={n_tsne}, perplexity={perp}")
tsne_out = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca").fit_transform(tsne_emb)
tsne_type_arr = np.array(tsne_type)
tsne_lab_arr  = np.array(tsne_lab)

fig8, axes8 = plt.subplots(1, 2, figsize=(16, 7))
for split, col, mk in [("seen", "#3498db", "o"), ("unseen", "#e74c3c", "^")]:
    mask = tsne_type_arr == split
    axes8[0].scatter(
        tsne_out[mask, 0], tsne_out[mask, 1], c=col,
        s=10 if split == "seen" else 18,
        alpha=0.4 if split == "seen" else 0.65,
        marker=mk, label=f"{split.capitalize()} ({mask.sum()})")
axes8[0].legend(fontsize=10)
axes8[0].axis("off")
axes8[0].set_title("t-SNE: Seen (●) vs Unseen (▲) Classes", fontweight="bold")

try:
    cmap_s = matplotlib.colormaps.get_cmap("Blues")
    cmap_u = matplotlib.colormaps.get_cmap("Reds")
except AttributeError:
    cmap_s = plt.get_cmap("Blues")
    cmap_u = plt.get_cmap("Reds")

cls_to_si = {c: i for i, c in enumerate(seen_classes)}
cls_to_ui = {c: i for i, c in enumerate(unseen_classes)}
for cls in seen_classes:
    mask = tsne_lab_arr == cls
    if not mask.sum():
        continue
    axes8[1].scatter(tsne_out[mask, 0], tsne_out[mask, 1],
                     c=[cmap_s(cls_to_si[cls] / S_n + 0.15)], alpha=0.28, s=8)
for cls in unseen_classes:
    mask = tsne_lab_arr == cls
    if not mask.sum():
        continue
    lbl = cls.replace("___", " - ").replace("_", " ")[:28]
    axes8[1].scatter(tsne_out[mask, 0], tsne_out[mask, 1],
                     c=[cmap_u(cls_to_ui[cls] / len(unseen_classes) + 0.15)],
                     alpha=0.7, s=20, marker="^", label=lbl)
handles, lbls = axes8[1].get_legend_handles_labels()
axes8[1].legend(handles, lbls, fontsize=6.5, loc="lower right",
                ncol=2, title="Unseen Classes (▲)")
axes8[1].set_title("Per-class Colouring (Blue = Seen, Red▲ = Unseen)", fontweight="bold")
axes8[1].axis("off")
fig8.suptitle(
    f"t-SNE Visualisation of DINOv2-Large Embeddings  (n={n_tsne}, perplexity={perp}, seed=42)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig8, "figure8_tsne.png")


def reliability_diag(probs, yt, title, ax, n_bins=15):
    conf = probs.max(1)
    pred = all_cls_arr[probs.argmax(1)]
    corr = (pred == np.asarray(yt)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ba   = []
    bc   = []
    bn   = []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        ba.append(corr[m].mean() if m.sum() > 0 else 0)
        bc.append(conf[m].mean() if m.sum() > 0 else (bins[i] + bins[i + 1]) / 2)
        bn.append(int(m.sum()))
    ece = sum(abs(a - c) * n for a, c, n in zip(ba, bc, bn)) / max(sum(bn), 1)
    mid = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Perfect calibration")
    ax.bar(mid, ba, width=1 / n_bins, alpha=0.7, color="#3498db",
           edgecolor="k", lw=0.5, label="Accuracy")
    ax.bar(mid, bc, width=1 / n_bins, alpha=0.25, color="red",
           edgecolor="k", lw=0.5, label="Confidence")
    ax.set_title(f"{title}\nECE = {ece*100:.2f}%", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")


fig_ece, axes_ece = plt.subplots(1, 2, figsize=(10, 5))
reliability_diag(fs_cal, s_labels_np,   "Seen Classes — Post-calibration",   axes_ece[0])
reliability_diag(fu_cal, u_labels_test, "Unseen Classes — Post-calibration", axes_ece[1])
fig_ece.suptitle("ECE Reliability Diagrams — Temperature Calibration (Validation Set)",
                 fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig_ece, "figure9_ece_reliability.png")


section("[16] Summary")

if RUN_MULTI_SEED and len(multi_seed_H3) > 1:
    multi_str = (f"3-shot H: {np.mean(multi_seed_H3)*100:.2f}% +/- "
                 f"{np.std(multi_seed_H3)*100:.2f}%  |  "
                 f"5-shot H: {np.mean(multi_seed_H5)*100:.2f}% +/- "
                 f"{np.std(multi_seed_H5)*100:.2f}%  ({len(multi_seed_H3)} seeds)")
else:
    multi_str = f"3-shot H={H3_p*100:.2f}%  5-shot H={H5_p*100:.2f}%  (seed=42)"

if RUN_MULTI_SEED and len(multi_seed_H3_c) > 1:
    multi_str_c = (f"3-shot H: {np.mean(multi_seed_H3_c)*100:.2f}% +/- "
                   f"{np.std(multi_seed_H3_c)*100:.2f}%  |  "
                   f"5-shot H: {np.mean(multi_seed_H5_c)*100:.2f}% +/- "
                   f"{np.std(multi_seed_H5_c)*100:.2f}%")
else:
    multi_str_c = f"3-shot H={H3_c*100:.2f}%  5-shot H={H5_c*100:.2f}%  (seed=42)"

local_only_str = "\n".join(
    f"    {cname:<12} local-only H={r['H']:.2f}%  (federated: {H_flp2*100:.2f}%)"
    for cname, r in local_only_results.items()
)

dp_line_str = (
    f"    FL-BCVSA + DP (clip={DP_CLIP_NORM}, noise_mult={DP_NOISE_MULTIPLIER}): "
    f"H={H_dp*100:.2f}%  (utility cost of DP: {H_flp2*100-H_dp*100:.2f}pp on top of federation)"
    if H_dp is not None else
    "    DP-FedAvg not run (RUN_DP_FL=False)"
)

print(f"""
{'='*72}
  BCVSA — COMPLETE RESULTS
  Total runtime: {elapsed()}
  Calibration protocol: held-out labeled validation split (seed=42)
{'='*72}
  PROPOSED METHOD
    0-shot : ZSL={acc_zsl_p*100:.2f}% (N={len(u_lab_all)})  S={acc_s_p*100:.2f}%  U={acc_u_p*100:.2f}%  H={H_p*100:.2f}%
    {multi_str}
    F1  Seen={met_p_s["f1"]*100:.2f}%   Unseen={met_p_u["f1"]*100:.2f}%
    ECE Seen={ece_p_s*100:.2f}%   Unseen={ece_p_u*100:.2f}%
{'-'*72}
  ABLATION A — CLIP Only
    ZSL={best_zsl_a*100:.2f}%  S={acc_s_a*100:.2f}%  U={acc_u_a*100:.2f}%  H={H_a*100:.2f}%
    ECE Seen={ece_a_s*100:.2f}%   Unseen={ece_a_u*100:.2f}%
{'-'*72}
  ABLATION B — DINOv2 Probe (Supervised)
    Acc={met_b["acc"]*100:.2f}%  F1={met_b["f1"]*100:.2f}%
{'-'*72}
  ABLATION C — Alignment Head
    0-shot ZSL={best_zsl_c*100:.2f}%  S={acc_s_c*100:.2f}%  U={acc_u_c*100:.2f}%  H={H_c*100:.2f}%
    {multi_str_c}
    ECE Seen={ece_c_s*100:.2f}%   Unseen={ece_c_u*100:.2f}%
{'-'*72}
  FEDERATED LEARNING (FL-BCVSA)
    Reused centralized calibration : H={H_flp*100:.2f}%
    Self-calibrated (report this)  : H={H_flp2*100:.2f}%  (alpha={af_fl} gamma={gf_fl:.4f} T={Tf_fl})
    Checkpoint selected by         : validation H (best val H={fl_bestH_p*100:.2f}%)
    Privacy cost                   : {H_p*100-H_flp2*100:.2f} pp
    Local-only baselines (no federation), for comparison:
{local_only_str}
{dp_line_str}
{'-'*72}
  SEVERITY ESTIMATION (unseen-side metrics reported on the held-out
  evaluation split only; per-crop centroids drawn from seen data where
  available, otherwise from the unseen calibration split)
    Healthy mean={h_vals.mean():.4f}  Diseased mean={d_vals.mean():.4f}  Ratio={ratio:.2f}x
    Grading acc={grade_acc*100:.0f}%
{'-'*72}
  CALIBRATION PARAMETERS
    alpha={af}  gamma={gf:.4f}  T={Tf}
    ZSL N={len(u_lab_all)}  GZSL N_test={N_utest}
{'-'*72}
  SAVED FIGURES ({len(saved_figs)} total)
""")
for fp in saved_figs:
    print(f"    {os.path.basename(fp)}")
print(f"""
  Results directory: {RESULTS_DIR}
{'='*72}
""")