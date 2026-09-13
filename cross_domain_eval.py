import os, glob, json, pickle, time, warnings
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
warnings.filterwarnings("ignore")

import random
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(GLOBAL_SEED)
try:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(GLOBAL_SEED)
except Exception: pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

t0 = time.time()
def elapsed():    return f"{(time.time()-t0)/60:.1f} min"
def section(msg): print(f"\n{'='*70}\n{msg}  ({elapsed()})\n{'='*70}")
def tick(msg):    print(f"  [{elapsed()}] {msg}")

OUTPUT_DIR = r"E:\VS Code\Projects\Thesis\TP002\PlantF\cross_eval_v2"

DEVICE = torch.device("xpu"  if (hasattr(torch, "xpu") and torch.xpu.is_available()) else
         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device: {DEVICE}")



CLIP_BATCH        = 64
DINO_BATCH        = 512
EPOCHS_PROBE      = 200
LR_PROBE          = 3e-4
CAL_SPLIT_SEED    = 42

T_CLIPS       = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
T_PROBE_GRID  = [0.5, 1.0, 1.5, 2.0, 3.0]
ALPHAS        = np.round(np.linspace(0, 1, 11), 2)
FEW_SHOT_KS   = [3, 5]


RUN_MULTI_SEED = True
MULTI_SEEDS    = [42, 123, 456, 789, 1000]

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
    parts = cls.split("___"); crop_k = parts[0]
    crop  = crop_k.replace("_", " ").lower()
    cond  = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
    desc  = CROP_DESC.get(crop_k, "")
    clean = cls.replace("___", " ").replace("_", " ")
    is_h  = "healthy" in cond.lower()
    if is_h:
        return [f"a high-resolution photo of a healthy {clean} plant leaf",
                f"a healthy {clean} leaf in natural lighting",
                f"a close-up image of a disease-free {clean} leaf",
                f"a pristine leaf of {clean}",
                f"healthy {crop} leaf, {desc}",
                f"close-up photo of healthy {crop} plant leaf",
                f"vibrant normal {crop} leaf without symptoms, {desc}",
                f"undamaged {crop} foliage in natural condition",
                f"high resolution photo of {crop} plant leaf",
                f"a well-cared {clean} leaf with no disease"]
    return [f"a high-resolution photo of a {clean} diseased plant leaf",
            f"a close-up image of {clean} disease symptoms",
            f"a plant leaf affected by {clean}",
            f"a leaf exhibiting signs of {clean} disease",
            f"a clear image of a {clean} infected leaf",
            f"high resolution photo of {crop} plant leaf",
            f"close-up photo of {crop} plant leaf showing {cond}",
            f"{crop} plant leaf with {cond} disease, {desc}",
            f"pathology image of {crop} {cond}",
            f"infected {crop} foliage with {cond} symptoms"]


def softmax_np(x, T=1.0):
    x = np.asarray(x, dtype=np.float64) / T
    e = np.exp(x - x.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)

def pacc(p, y): return float((np.asarray(p) == np.asarray(y)).mean())

def full_metrics(y_true, y_pred, labels):
    return {
        "acc":  pacc(y_pred, y_true),
        "f1":   f1_score(  y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "prec": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "rec":  recall_score(   y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }

def compute_ece(probs, y_true, label_arr, n_bins=15):
    conf = probs.max(1)
    pred = label_arr[probs.argmax(1)]
    corr = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum():
            total += abs(corr[m].mean() - conf[m].mean()) * m.sum()
    return total / max(len(y_true), 1)

def error_analysis(pred_arr, true_arr, class_list):
    rows = []
    for cls in class_list:
        mask = np.asarray(true_arr) == cls
        if not mask.sum(): continue
        n = int(mask.sum()); preds = np.asarray(pred_arr)[mask]
        correct = int((preds == cls).sum()); wrong_p = preds[preds != cls]
        if len(wrong_p):
            vals, cnts = np.unique(wrong_p, return_counts=True)
            tw = vals[cnts.argmax()]; tw_n = int(cnts.max())
        else:
            tw = "None"; tw_n = 0
        pred_all = np.asarray(pred_arr); true_all = np.asarray(true_arr)
        tp = int(((pred_all == cls) & (true_all == cls)).sum())
        fp = int(((pred_all == cls) & (true_all != cls)).sum())
        fn = int(((pred_all != cls) & (true_all == cls)).sum())
        prec = tp/(tp+fp) if (tp+fp) else 0.
        rec  = tp/(tp+fn) if (tp+fn) else 0.
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.
        rows.append({"Class": cls, "N": n, "Correct": correct,
                     "Accuracy_%": round(correct / n * 100, 2),
                     "F1_%": round(f1*100, 2),
                     "Top_Wrong_Class": tw, "Top_Wrong_N": tw_n,
                     "Top_Wrong_%": round(tw_n / n * 100, 2)})
    return pd.DataFrame(rows).sort_values("Accuracy_%")

def hd_analysis(pred_arr, true_arr, class_list, split):
    rows = []
    for cls in class_list:
        mask = np.asarray(true_arr) == cls
        if not mask.sum(): continue
        n = int(mask.sum()); p = np.asarray(pred_arr)[mask]
        correct = int((p == cls).sum()); as_h = int(sum("Healthy" in x for x in p))
        rows.append({"Split": split, "Class": cls,
                     "Type": "Healthy" if "Healthy" in cls else "Diseased",
                     "N": n, "Accuracy_%": round(correct / n * 100, 2),
                     "Predicted_Healthy_%": round(as_h / n * 100, 2),
                     "Predicted_Diseased_%": round((n - as_h) / n * 100, 2)})
    return pd.DataFrame(rows)

def wrong_pairs(pred_arr, true_arr):
    rows = []
    for t, p in zip(true_arr, pred_arr):
        if t != p: rows.append({"True": t, "Predicted": p})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).groupby(["True", "Predicted"]).size().reset_index(name="Count")
    df["True_Label"] = df["True"].str.replace("___", " — ").str.replace("_", " ")
    df["Pred_Label"] = df["Predicted"].str.replace("___", " — ").str.replace("_", " ")
    return df.sort_values("Count", ascending=False)


def run_direction(direction_dir, direction_label, source_name, target_name):

    SAVE_DIR        = os.path.join(direction_dir, "metadata")
    RESULTS_DIR     = os.path.join(direction_dir, "results")
    SOURCE_DIR      = os.path.join(direction_dir, "source")
    TRAIN_DIR       = os.path.join(SOURCE_DIR, "train")
    VAL_DIR         = os.path.join(SOURCE_DIR, "val")
    SEEN_TEST_DIR   = os.path.join(SOURCE_DIR, "test")
    TARGET_DIR      = os.path.join(direction_dir, "target", "unseen_test")
    BUNDLE_PATH     = os.path.join(SAVE_DIR, "cross_bundle.pkl")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    saved_figs = []
    def savefig(fig, fname):
        path = os.path.join(RESULTS_DIR, fname)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_figs.append(path)
        print(f"  Saved: {fname}")

    section(f"[{direction_label}] Loading Feature Bundle")
    with open(BUNDLE_PATH, "rb") as f: bundle = pickle.load(f)

    meta = pd.DataFrame(bundle["metadata"])
    meta.index = range(len(meta)); meta["pos_idx"] = meta.index
    if "class_name" not in meta.columns and "class" in meta.columns:
        meta = meta.rename(columns={"class": "class_name"})
    if "type" not in meta.columns:
        meta["type"] = "seen"
        meta.loc[meta["split"].str.contains("unseen", case=False, na=False), "type"] = "unseen"

    dino_emb   = F.normalize(torch.tensor(bundle["embeddings"], dtype=torch.float32), dim=-1)
    train_meta = meta[(meta["type"] == "seen") & (meta["split"] == "train")].copy()
    val_meta   = meta[(meta["type"] == "seen") & (meta["split"] == "val")].copy()
    stest_meta = meta[(meta["type"] == "seen") & (meta["split"] == "test")].copy()
    ttest_meta = meta[meta["type"] == "unseen"].copy()   # target-domain images

    source_classes = sorted([d for d in os.listdir(TRAIN_DIR)
                              if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    target_classes = sorted([d for d in os.listdir(TARGET_DIR)
                              if os.path.isdir(os.path.join(TARGET_DIR, d))])
    not_shared = set(target_classes) - set(source_classes)
    if not_shared:
        raise RuntimeError(
            f"Target classes not present in source taxonomy: {not_shared}. "
            "This evaluation assumes target_classes is a subset of "
            "source_classes -- if that's no longer true, this is genuine "
            "zero-shot again and needs the disjoint-class GZSL protocol "
            "instead of this one.")

    cls2id      = {c: i for i, c in enumerate(source_classes)}
    source_arr  = np.array(source_classes)
    S_n         = len(source_classes)
    train_meta["seen_id"] = train_meta["class_name"].map(cls2id)
    val_meta["seen_id"]   = val_meta["class_name"].map(cls2id)

    print(f"  Source classes: {S_n}  |  Target classes: {len(target_classes)} "
          f"(subset of source: {set(target_classes) <= set(source_classes)})")
    print(f"  Source: {source_name}  |  Target: {target_name}")
    print(f"  Images — source: {len(train_meta)+len(val_meta)+len(stest_meta)}  "
          f"target: {len(ttest_meta)}")


    section(f"[{direction_label}] Calibration / Evaluation Split of Target Domain")
    # 50/50 per class: calibration half tunes (T_probe, T_clip, alpha) and the
    # ECE temperature; evaluation half is untouched until final reporting.
    np.random.seed(CAL_SPLIT_SEED)
    t_meta_arr = np.arange(len(ttest_meta))
    t_cls_arr  = np.array(ttest_meta["class_name"].tolist())
    cal_mask   = np.zeros(len(ttest_meta), dtype=bool)
    eval_mask  = np.zeros(len(ttest_meta), dtype=bool)
    for cls in target_classes:
        pos  = t_meta_arr[t_cls_arr == cls]
        perm = np.random.permutation(len(pos))
        half = len(pos) // 2
        cal_mask[pos[perm[:half]]]  = True
        eval_mask[pos[perm[half:]]] = True
    N_cal  = int(cal_mask.sum())
    N_eval = int(eval_mask.sum())
    print(f"  Calibration split : {N_cal} images")
    print(f"  Evaluation split  : {N_eval} images")


    section(f"[{direction_label}] CLIP Text Prototype Generation")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").float().to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    IPEX = False
    try:
        import intel_extension_for_pytorch as ipex
        clip_model = ipex.optimize(clip_model); IPEX = True; print("  IPEX enabled")
    except Exception: print("  IPEX not available")

    text_cache = os.path.join(SAVE_DIR, "clip_text_prototypes.pt")
    if os.path.exists(text_cache):
        all_text_cpu = torch.load(text_cache, weights_only=False)
        print("  Loaded from cache")
    else:
        text_features = {}
        with torch.no_grad():
            for cls in tqdm(source_classes, desc="  Generating text prototypes"):
                prompts = make_prompts(cls)
                inp = clip_processor(text=prompts, return_tensors="pt",
                                     padding=True, truncation=True).to(DEVICE)
                f   = F.normalize(clip_model.get_text_features(**inp).float(), dim=-1)
                text_features[cls] = F.normalize(f.mean(0), dim=-1).cpu()
        all_text_cpu = torch.stack([text_features[c] for c in source_classes])
        torch.save(all_text_cpu, text_cache)
    all_text_np = all_text_cpu.numpy()
    print(f"  Text prototypes ready. Shape: {all_text_cpu.shape}")


    section(f"[{direction_label}] CLIP Image Feature Extraction")
    def extract_clip(base_dir, cls_list, cache_path, label):
        if os.path.exists(cache_path):
            tick(f"Cache loaded: {os.path.basename(cache_path)}")
            d = torch.load(cache_path, weights_only=False)
            return d["feat"], np.array(d["labels"]), d["class_filenames"]
        all_f, all_l, all_cf = [], [], []
        for cls in tqdm(cls_list, desc=f"  {label}"):
            cdir = os.path.join(base_dir, cls)
            if not os.path.isdir(cdir): continue
            paths = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
            for i in range(0, len(paths), CLIP_BATCH):
                bp = paths[i:i + CLIP_BATCH]
                imgs = []
                for p in bp:
                    try:    imgs.append(Image.open(p).convert("RGB"))
                    except: imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
                inp = clip_processor(images=imgs, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    feat = F.normalize(clip_model.get_image_features(**inp).float(), dim=-1)
                all_f.append(feat.cpu()); all_l.extend([cls] * len(bp))
                all_cf.extend([f"{cls}/{os.path.basename(p)}" for p in bp])
                if DEVICE.type == "xpu": torch.xpu.empty_cache()
        fc = torch.cat(all_f)
        torch.save({"feat": fc, "labels": np.array(all_l), "class_filenames": all_cf}, cache_path)
        return fc, np.array(all_l), all_cf

    clip_t_all, t_lab_all, t_cf_all = extract_clip(
        TARGET_DIR, target_classes,
        os.path.join(SAVE_DIR, "clip_features_target.pt"), "Target domain")
    clip_v, v_labels, _ = extract_clip(
        VAL_DIR, source_classes,
        os.path.join(SAVE_DIR, "clip_features_val.pt"), "Source val")
    clip_s, s_labels, _ = extract_clip(
        SEEN_TEST_DIR, source_classes,
        os.path.join(SAVE_DIR, "clip_features_test.pt"), "Source test")
    del clip_model

    t_cf_to_idx = {cf: i for i, cf in enumerate(t_cf_all)}
    t_lab_all_np = np.array(t_lab_all)
    clip_t_cal  = clip_t_all[cal_mask]
    clip_t_eval = clip_t_all[eval_mask]
    t_labels_cal  = t_lab_all_np[cal_mask]
    t_labels_eval = t_lab_all_np[eval_mask]
    s_labels_np = np.array(s_labels)
    v_labels_np = np.array(v_labels)

    clip_t_raw_cal  = (clip_t_cal  @ all_text_cpu.T).numpy()
    clip_t_raw_eval = (clip_t_eval @ all_text_cpu.T).numpy()
    clip_s_raw      = (clip_s @ all_text_cpu.T).numpy()
    clip_v_raw      = (clip_v @ all_text_cpu.T).numpy()
    tick("CLIP cosine similarities computed")


    section(f"[{direction_label}] Training DINOv2 Linear Probe ({EPOCHS_PROBE} epochs)")
    class LinearProbe(nn.Module):
        def __init__(self, out=S_n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512,  256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, out))
        def forward(self, x): return self.net(x)

    probe_path_best = os.path.join(SAVE_DIR, "probe_best.pt")
    probe = LinearProbe().to(DEVICE)
    opt_p = torch.optim.AdamW(probe.parameters(), lr=LR_PROBE, weight_decay=1e-3)
    sch_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=EPOCHS_PROBE, eta_min=1e-6)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    if IPEX:
        try: probe, opt_p = ipex.optimize(probe, optimizer=opt_p, level="O1")
        except Exception: pass

    tr_idx = train_meta["pos_idx"].values
    tr_lab = torch.tensor(train_meta["seen_id"].values, dtype=torch.long)
    best_acc = 0.

    if os.path.exists(probe_path_best):
        tick("Probe checkpoint found — loading")
        probe.load_state_dict(torch.load(probe_path_best, map_location=DEVICE, weights_only=False))
        probe.eval()
        with torch.no_grad():
            vv = dino_emb[val_meta["pos_idx"].values].to(DEVICE)
            pv = [source_classes[i] for i in probe(vv).argmax(1).cpu().numpy()]
            best_acc = accuracy_score(val_meta["class_name"].tolist(), pv)
        print(f"  Loaded probe validation accuracy: {best_acc*100:.2f}%")
    else:
        for epoch in range(EPOCHS_PROBE):
            probe.train()
            perm = np.random.permutation(len(tr_idx))
            for i in range(0, len(perm), DINO_BATCH):
                idx = tr_idx[perm[i:i+DINO_BATCH]]; v = dino_emb[idx].to(DEVICE)
                l = tr_lab[perm[i:i+DINO_BATCH]].to(DEVICE)
                opt_p.zero_grad(); ce_fn(probe(v), l).backward(); opt_p.step()
            sch_p.step()
            if (epoch + 1) % 20 == 0:
                probe.eval()
                with torch.no_grad():
                    vv = dino_emb[val_meta["pos_idx"].values].to(DEVICE)
                    pv = [source_classes[i] for i in probe(vv).argmax(1).cpu().numpy()]
                    vacc = accuracy_score(val_meta["class_name"].tolist(), pv)
                if vacc > best_acc:
                    best_acc = vacc
                    torch.save(probe.state_dict(), probe_path_best)
                print(f"  Epoch {epoch+1:3d}/{EPOCHS_PROBE}  Val Accuracy={vacc*100:.2f}%  ({elapsed()})")
        probe.load_state_dict(torch.load(probe_path_best, map_location=DEVICE, weights_only=False))
        probe.eval()
    print(f"  Best validation accuracy: {best_acc*100:.2f}%")

    def get_probe_sm(idx_arr, T=1.0):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx_arr), DINO_BATCH):
                idx = idx_arr[i:i+DINO_BATCH]
                out.append(F.softmax(probe(dino_emb[idx].to(DEVICE)) / T, dim=-1).cpu().numpy())
        return np.vstack(out)

    def get_probe_logits(idx_arr):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx_arr), DINO_BATCH):
                idx = idx_arr[i:i+DINO_BATCH]
                out.append(probe(dino_emb[idx].to(DEVICE)).cpu().numpy())
        return np.vstack(out)


    section(f"[{direction_label}] In-Domain and Single-Branch Baselines")

    probe_logit_s = get_probe_logits(stest_meta["pos_idx"].values)
    pred_in_domain = source_arr[probe_logit_s.argmax(1)]
    met_in_domain  = full_metrics(s_labels_np, pred_in_domain, source_classes)
    print(f"  In-domain (probe on {source_name} test): "
          f"acc={met_in_domain['acc']*100:.2f}%  F1={met_in_domain['f1']*100:.2f}%")

    probe_logit_t_eval = get_probe_logits(ttest_meta["pos_idx"].values[eval_mask])
    pred_cross_probe = source_arr[probe_logit_t_eval.argmax(1)]
    met_cross_probe  = full_metrics(t_labels_eval, pred_cross_probe, target_classes)
    print(f"  Cross-domain, probe only (no CLIP): "
          f"acc={met_cross_probe['acc']*100:.2f}%  F1={met_cross_probe['f1']*100:.2f}%")

    pred_cross_clip = source_arr[clip_t_raw_eval.argmax(1)]
    met_cross_clip  = full_metrics(t_labels_eval, pred_cross_clip, target_classes)
    print(f"  Cross-domain, CLIP only (zero-shot, no training): "
          f"acc={met_cross_clip['acc']*100:.2f}%  F1={met_cross_clip['f1']*100:.2f}%")


    section(f"[{direction_label}] Cross-Domain Fusion Search (alpha x T_probe x T_clip)")

    probe_logit_t_cal = get_probe_logits(ttest_meta["pos_idx"].values[cal_mask])

    best = {"acc": -1., "T_probe": 1.0, "T_clip": 0.07, "alpha": 0.5}
    for T_probe in tqdm(T_PROBE_GRID, desc="  T_probe"):
        probe_sm_cal_tp = softmax_np(probe_logit_t_cal, T_probe)
        for T_clip in T_CLIPS:
            clip_sm_cal = softmax_np(clip_t_raw_cal, T_clip)
            for alpha in ALPHAS:
                fused = alpha * probe_sm_cal_tp + (1 - alpha) * clip_sm_cal
                acc = pacc(source_arr[fused.argmax(1)], t_labels_cal)
                if acc > best["acc"]:
                    best.update({"acc": acc, "T_probe": float(T_probe),
                                 "T_clip": float(T_clip), "alpha": float(alpha)})

    T_probe_f, T_clip_f, alpha_f = best["T_probe"], best["T_clip"], best["alpha"]
    print(f"  Best calibration-split accuracy: {best['acc']*100:.2f}%  "
          f"(T_probe={T_probe_f}  T_clip={T_clip_f}  alpha={alpha_f})")


    section(f"[{direction_label}] Cross-Domain Evaluation (Held-Out Target Split)")
    probe_sm_eval_tp = softmax_np(probe_logit_t_eval, T_probe_f)
    clip_sm_eval     = softmax_np(clip_t_raw_eval, T_clip_f)
    fused_eval       = alpha_f * probe_sm_eval_tp + (1 - alpha_f) * clip_sm_eval
    pred_fusion      = source_arr[fused_eval.argmax(1)]
    met_fusion       = full_metrics(t_labels_eval, pred_fusion, target_classes)
    print(f"  Cross-domain fusion (BCVSA-style, 0-shot): "
          f"acc={met_fusion['acc']*100:.2f}%  F1={met_fusion['f1']*100:.2f}%")


    label_idx_cal = np.array([cls2id[c] for c in t_labels_cal])
    fused_cal = alpha_f * softmax_np(probe_logit_t_cal, T_probe_f) + \
                (1 - alpha_f) * softmax_np(clip_t_raw_cal, T_clip_f)
    pseudo_logit_cal = np.log(np.clip(fused_cal, 1e-9, None))

    def nll(log_T):
        T = float(np.exp(log_T))
        sc = pseudo_logit_cal / T
        sh = sc - sc.max(1, keepdims=True)
        lsm = np.log(np.exp(sh).sum(1, keepdims=True))
        return -(sh[np.arange(len(label_idx_cal)), label_idx_cal] - lsm.squeeze()).mean()

    res_cal = minimize_scalar(nll, bounds=(-4., 2.), method="bounded", options={"xatol": 1e-4})
    T_cal   = float(np.exp(res_cal.x))
    pseudo_logit_eval = np.log(np.clip(fused_eval, 1e-9, None))
    cal_exp = np.exp(pseudo_logit_eval / T_cal - (pseudo_logit_eval / T_cal).max(1, keepdims=True))
    fused_eval_calibrated = cal_exp / cal_exp.sum(1, keepdims=True)
    ece_fusion = compute_ece(fused_eval_calibrated, t_labels_eval, source_arr)
    print(f"  ECE (fusion, calibrated, T_cal={T_cal:.4f}): {ece_fusion*100:.2f}%")


    section(f"[{direction_label}] Few-Shot Cross-Domain Adaptation")
    clip_model2     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").float().to(DEVICE)
    clip_processor2 = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model2.eval()
    if IPEX:
        try: clip_model2 = ipex.optimize(clip_model2)
        except Exception: pass

    def few_shot_adapt(k, seed=42):
        np.random.seed(seed)
        atf = all_text_cpu.clone()
        support_global = set()
        for cls in target_classes:
            paths = sorted(glob.glob(os.path.join(TARGET_DIR, cls, "*.jpg")))
            if len(paths) < k: continue
            chosen = list(np.random.choice(paths, k, replace=False))
            imgs = [Image.open(p).convert("RGB") for p in chosen]
            inp  = clip_processor2(images=imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                vf = F.normalize(clip_model2.get_image_features(**inp).float(), dim=-1).cpu()
            atf[cls2id[cls]] = F.normalize(0.5*vf.mean(0) + 0.5*all_text_cpu[cls2id[cls]], dim=-1)
            for p in chosen:
                key = f"{cls}/{os.path.basename(p)}"
                if key in t_cf_to_idx: support_global.add(t_cf_to_idx[key])

        eval_global_idx = np.where(eval_mask)[0]
        support_in_eval = support_global & set(eval_global_idx.tolist())
        eval_mask_fs = np.ones(N_eval, dtype=bool)
        if support_in_eval:
            g2l = {g: i for i, g in enumerate(eval_global_idx)}
            for g in support_in_eval: eval_mask_fs[g2l[g]] = False

        clip_t_fs_eval = (clip_t_eval @ atf.T).numpy()
        clip_sm_fs = softmax_np(clip_t_fs_eval, T_clip_f)
        fused_fs = alpha_f * probe_sm_eval_tp + (1 - alpha_f) * clip_sm_fs
        pred_fs  = source_arr[fused_fs[eval_mask_fs].argmax(1)]
        true_fs  = t_labels_eval[eval_mask_fs]
        return full_metrics(true_fs, pred_fs, target_classes)

    met_3shot = few_shot_adapt(3)
    met_5shot = few_shot_adapt(5)
    print(f"  3-shot: acc={met_3shot['acc']*100:.2f}%  F1={met_3shot['f1']*100:.2f}%")
    print(f"  5-shot: acc={met_5shot['acc']*100:.2f}%  F1={met_5shot['f1']*100:.2f}%")


    multi_seed_acc3 = [met_3shot["acc"]]
    multi_seed_acc5 = [met_5shot["acc"]]
    multi_seed_f13  = [met_3shot["f1"]]
    multi_seed_f15  = [met_5shot["f1"]]
    if RUN_MULTI_SEED:
        print(f"\n  Multi-seed robustness check ({MULTI_SEEDS})...")
        for s in MULTI_SEEDS:
            if s == 42:
                continue
            r3 = few_shot_adapt(3, seed=s)
            r5 = few_shot_adapt(5, seed=s)
            multi_seed_acc3.append(r3["acc"]); multi_seed_f13.append(r3["f1"])
            multi_seed_acc5.append(r5["acc"]); multi_seed_f15.append(r5["f1"])
            print(f"    seed={s}  3-shot acc={r3['acc']*100:.2f}%  "
                  f"5-shot acc={r5['acc']*100:.2f}%")
        print(f"  3-shot acc: {np.mean(multi_seed_acc3)*100:.2f}% +/- "
              f"{np.std(multi_seed_acc3)*100:.2f}%  ({len(multi_seed_acc3)} seeds)")
        print(f"  5-shot acc: {np.mean(multi_seed_acc5)*100:.2f}% +/- "
              f"{np.std(multi_seed_acc5)*100:.2f}%  ({len(multi_seed_acc5)} seeds)")

    del clip_model2


    section(f"[{direction_label}] Error Analysis")
    df_err = error_analysis(pred_fusion, t_labels_eval, target_classes)
    df_hd  = hd_analysis(pred_fusion, t_labels_eval, target_classes, "cross_domain_fusion")
    df_wp  = wrong_pairs(pred_fusion, t_labels_eval)

    print(f"\n  Cross-Domain Fusion — Per-Class Accuracy ({source_name} → {target_name})")
    print(f"  {'Class':<45} {'N':>6} {'Acc%':>7} {'F1%':>6} {'Top Confusion':<35} {'%':>6}")
    for _, r in df_err.iterrows():
        cls_s = r["Class"].replace("___", " — ").replace("_", " ")[:43]
        tw_s  = str(r["Top_Wrong_Class"]).replace("___", " — ").replace("_", " ")[:33]
        print(f"  {cls_s:<45} {r['N']:>6} {r['Accuracy_%']:>6.1f}% {r['F1_%']:>5.1f}%  "
              f"{tw_s:<35} {r['Top_Wrong_%']:>5.1f}%")


    section(f"[{direction_label}] Saving Results")
    master_rows = [
        {"Method": "CLIP only (zero-shot)",  "Setting": "cross-domain",
         "Acc%": round(met_cross_clip["acc"]*100,2),  "F1%": round(met_cross_clip["f1"]*100,2)},
        {"Method": "DINOv2 probe only",       "Setting": "cross-domain",
         "Acc%": round(met_cross_probe["acc"]*100,2), "F1%": round(met_cross_probe["f1"]*100,2)},
        {"Method": "DINOv2 probe only",       "Setting": "in-domain",
         "Acc%": round(met_in_domain["acc"]*100,2),   "F1%": round(met_in_domain["f1"]*100,2)},
        {"Method": "BCVSA fusion (0-shot)",   "Setting": "cross-domain",
         "Acc%": round(met_fusion["acc"]*100,2),       "F1%": round(met_fusion["f1"]*100,2)},
        {"Method": "BCVSA fusion (3-shot)",   "Setting": "cross-domain",
         "Acc%": round(met_3shot["acc"]*100,2),        "F1%": round(met_3shot["f1"]*100,2)},
        {"Method": "BCVSA fusion (5-shot)",   "Setting": "cross-domain",
         "Acc%": round(met_5shot["acc"]*100,2),        "F1%": round(met_5shot["f1"]*100,2)},
    ]
    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(os.path.join(RESULTS_DIR, "cross_domain_results.csv"), index=False)
    df_err.to_csv(os.path.join(RESULTS_DIR, "error_analysis_target.csv"), index=False)
    df_hd.to_csv(os.path.join(RESULTS_DIR, "healthy_vs_diseased_target.csv"), index=False)
    df_wp.to_csv(os.path.join(RESULTS_DIR, "misclassification_pairs_target.csv"), index=False)

    if RUN_MULTI_SEED and len(multi_seed_acc3) > 1:
        few_shot_str = (f"mean+/-std ({len(multi_seed_acc3)} seeds): "
                        f"3-shot acc={np.mean(multi_seed_acc3)*100:.2f}+/-{np.std(multi_seed_acc3)*100:.2f}  "
                        f"5-shot acc={np.mean(multi_seed_acc5)*100:.2f}+/-{np.std(multi_seed_acc5)*100:.2f}")
    else:
        few_shot_str = f"3-shot acc={met_3shot['acc']*100:.2f}%  5-shot acc={met_5shot['acc']*100:.2f}%  (seed=42)"

    summary = {
        "direction": f"{source_name} → {target_name}",
        "protocol":  "shared-taxonomy cross-domain accuracy, 50/50 target calibration/evaluation split, seed=42",
        "fusion_params": {"alpha": alpha_f, "T_probe": T_probe_f, "T_clip": T_clip_f, "T_cal": round(T_cal, 4)},
        "in_domain_probe":        met_in_domain,
        "cross_domain_clip_only": met_cross_clip,
        "cross_domain_probe_only": met_cross_probe,
        "cross_domain_fusion_0shot": {**met_fusion, "ece_%": round(ece_fusion*100, 2)},
        "cross_domain_fusion_3shot": met_3shot,
        "cross_domain_fusion_5shot": met_5shot,
        "cross_domain_fusion_3shot_multiseed": {
            "acc_mean_%": round(float(np.mean(multi_seed_acc3)) * 100, 2) if RUN_MULTI_SEED else None,
            "acc_std_%":  round(float(np.std(multi_seed_acc3)) * 100, 2) if RUN_MULTI_SEED else None,
            "f1_mean_%":  round(float(np.mean(multi_seed_f13)) * 100, 2) if RUN_MULTI_SEED else None,
            "f1_std_%":   round(float(np.std(multi_seed_f13)) * 100, 2) if RUN_MULTI_SEED else None,
            "n_seeds":    len(multi_seed_acc3) if RUN_MULTI_SEED else 1,
        },
        "cross_domain_fusion_5shot_multiseed": {
            "acc_mean_%": round(float(np.mean(multi_seed_acc5)) * 100, 2) if RUN_MULTI_SEED else None,
            "acc_std_%":  round(float(np.std(multi_seed_acc5)) * 100, 2) if RUN_MULTI_SEED else None,
            "f1_mean_%":  round(float(np.mean(multi_seed_f15)) * 100, 2) if RUN_MULTI_SEED else None,
            "f1_std_%":   round(float(np.std(multi_seed_f15)) * 100, 2) if RUN_MULTI_SEED else None,
            "n_seeds":    len(multi_seed_acc5) if RUN_MULTI_SEED else 1,
        },
        "few_shot_summary": few_shot_str,
        "n_source_classes": S_n, "n_target_classes": len(target_classes),
        "n_target_calibration": N_cal, "n_target_evaluation": N_eval,
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tick("Results saved")
    print(df_master.to_string(index=False))


    section(f"[{direction_label}] Generating Figures")

    # Fig 1: simplified architecture diagram (no bias-penalty term).
    fig1, ax1 = plt.subplots(figsize=(13, 6))
    ax1.set_xlim(0, 13); ax1.set_ylim(0, 6); ax1.axis("off")
    def draw_box(ax, x, y, w, h, label, sublabel="", color="#2563EB", tc="white", fs=9):
        ax.add_patch(plt.Rectangle((x,y), w, h, facecolor=color, edgecolor="black",
                                    linewidth=1.5, zorder=3))
        ax.text(x+w/2, y+h/2+(0.13 if sublabel else 0), label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        if sublabel:
            ax.text(x+w/2, y+h/2-0.22, sublabel, ha="center", va="center",
                    fontsize=fs-1.5, color=tc, zorder=4)
    def arr(ax, x1, y1, x2, y2, color="black"):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)

    draw_box(ax1, 0.3, 2.9, 1.5, 0.9, "Input Image", color="#374151")
    arr(ax1, 1.8, 3.6, 2.6, 4.4); arr(ax1, 1.8, 3.2, 2.6, 1.8)
    draw_box(ax1, 2.6, 4.0, 2.4, 0.9, "DINOv2-Large", "(frozen)", color="#2563EB")
    arr(ax1, 5.0, 4.45, 5.8, 4.45)
    draw_box(ax1, 5.8, 4.0, 2.3, 0.9, "Probe", f"trained on {source_name}", color="#0D9488")
    arr(ax1, 8.1, 4.45, 8.9, 4.05)
    draw_box(ax1, 2.6, 1.35, 2.4, 0.9, "CLIP ViT-L/14", "(frozen)", color="#EA580C")
    arr(ax1, 5.0, 1.8, 5.8, 1.8)
    draw_box(ax1, 5.8, 1.35, 2.3, 0.9, "Cosine Sim.", f"{S_n} class prototypes", color="#EA580C")
    arr(ax1, 8.1, 1.8, 8.9, 3.15)
    ax1.add_patch(plt.Rectangle((8.9, 2.4), 2.6, 2.0, facecolor="#F3F4F6",
                                  edgecolor="black", linewidth=1.2, zorder=3))
    ax1.text(10.2, 4.05, "Domain-Adaptive Fusion", fontsize=9.5, fontweight="bold",
             ha="center", color="#1F2937")
    ax1.text(10.2, 3.6, r"$\alpha\cdot\sigma(z_{probe}/T_p)+(1-\alpha)\cdot\sigma(z_{clip}/T_c)$",
             fontsize=8, ha="center", color="#1F2937")
    ax1.text(10.2, 3.1, f"α={alpha_f}  T_p={T_probe_f}  T_c={T_clip_f}",
             fontsize=8, ha="center", fontweight="bold", color="#DC2626")
    ax1.text(10.2, 2.7, "tuned on target calibration split only",
             fontsize=7, ha="center", color="#6B7280", style="italic")
    arr(ax1, 10.2, 2.4, 10.2, 1.9)
    draw_box(ax1, 8.9, 0.9, 2.6, 0.8, "Prediction",
             f"argmax over {S_n} classes", color="#16A34A", fs=8.5)
    ax1.legend(handles=[
        mpatches.Patch(color="#2563EB", label="DINOv2 (frozen)"),
        mpatches.Patch(color="#0D9488", label="Trainable probe"),
        mpatches.Patch(color="#EA580C", label="CLIP (frozen)"),
        mpatches.Patch(color="#16A34A", label="Output"),
    ], loc="lower left", fontsize=8, frameon=True)
    ax1.set_title(f"Cross-Domain BCVSA: {source_name} → {target_name}\n"
                  f"Probe trained on {source_name}; evaluated cross-domain on {target_name} "
                  f"({len(target_classes)} shared classes)",
                  fontsize=11.5, fontweight="bold", pad=8)
    plt.tight_layout()
    savefig(fig1, "figure1_architecture.png")

    # Fig 2: method comparison bar chart.
    methods_f2 = ["CLIP only\n(zero-shot)", "Probe only\n(cross-domain)",
                  "Probe only\n(in-domain)", "Fusion\n0-shot", "Fusion\n3-shot", "Fusion\n5-shot"]
    acc_f2 = [met_cross_clip["acc"]*100, met_cross_probe["acc"]*100, met_in_domain["acc"]*100,
              met_fusion["acc"]*100, met_3shot["acc"]*100, met_5shot["acc"]*100]
    f1_f2  = [met_cross_clip["f1"]*100, met_cross_probe["f1"]*100, met_in_domain["f1"]*100,
              met_fusion["f1"]*100, met_3shot["f1"]*100, met_5shot["f1"]*100]
    x2 = np.arange(len(methods_f2)); w2 = 0.35
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    b1 = ax2.bar(x2 - w2/2, acc_f2, w2, label="Accuracy %", color="#2563EB", alpha=0.88)
    b2 = ax2.bar(x2 + w2/2, f1_f2,  w2, label="Macro-F1 %", color="#16A34A", alpha=0.88)
    for bars in [b1, b2]:
        for rect in bars:
            v = rect.get_height()
            ax2.text(rect.get_x()+rect.get_width()/2, v+0.5, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x2); ax2.set_xticklabels(methods_f2, fontsize=8.5)
    ax2.set_ylim(0, 108); ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.set_title(f"Cross-Domain Method Comparison: {source_name} → {target_name}",
                  fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")
    ax2.axvline(x=2.5, color="gray", lw=1, linestyle="--", alpha=0.5)
    plt.tight_layout()
    savefig(fig2, "figure2_method_comparison.png")

    # Fig 3: alpha x T_probe accuracy heatmap at the best T_clip.
    heat = np.zeros((len(ALPHAS), len(T_PROBE_GRID)))
    clip_sm_cal_best = softmax_np(clip_t_raw_cal, T_clip_f)
    for i, a in enumerate(ALPHAS):
        for j, tp in enumerate(T_PROBE_GRID):
            probe_sm_tmp = softmax_np(probe_logit_t_cal, tp)
            fused_tmp = a*probe_sm_tmp + (1-a)*clip_sm_cal_best
            heat[i, j] = pacc(source_arr[fused_tmp.argmax(1)], t_labels_cal) * 100
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    im = ax3.imshow(heat, aspect="auto", origin="lower", cmap="RdYlGn",
                    vmin=heat.min(), vmax=heat.max())
    plt.colorbar(im, ax=ax3, label="Calibration-split Accuracy (%)", fraction=0.045, pad=0.03)
    ax3.set_xticks(range(len(T_PROBE_GRID))); ax3.set_xticklabels(T_PROBE_GRID, fontsize=8)
    ax3.set_yticks(range(len(ALPHAS)));       ax3.set_yticklabels([f"{a:.1f}" for a in ALPHAS], fontsize=8)
    ax3.set_xlabel("T_probe", fontsize=10); ax3.set_ylabel("alpha (probe weight)", fontsize=10)
    ai = list(ALPHAS).index(alpha_f) if alpha_f in ALPHAS else None
    ti = T_PROBE_GRID.index(T_probe_f) if T_probe_f in T_PROBE_GRID else None
    if ai is not None and ti is not None:
        ax3.plot(ti, ai, "r*", markersize=16, label=f"Optimal (T_clip={T_clip_f})")
        ax3.legend(fontsize=8)
    ax3.set_title(f"Fusion Search Landscape: {source_name} → {target_name}",
                  fontsize=10.5, fontweight="bold")
    plt.tight_layout()
    savefig(fig3, "figure3_fusion_heatmap.png")

    # Fig 4: in-domain (source) vs cross-domain (target) confusion matrices.
    def plot_cm(yt, yp, labs, title, ax):
        cm  = confusion_matrix(yt, yp, labels=labs)
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        short = [c.split("___")[-1].replace("_", " ")[:13] for c in labs]
        n = len(labs)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        for i in range(n):
            for j in range(n):
                if cmn[i, j] > 0.08:
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                            fontsize=5, color="white" if cmn[i,j] > 0.55 else "black")
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        return im
    fig4, axes4 = plt.subplots(1, 2, figsize=(20, 9))
    im1 = plot_cm(s_labels_np, pred_in_domain, source_classes,
                  f"In-Domain ({source_name}, probe only)", axes4[0])
    im2 = plot_cm(t_labels_eval, pred_fusion, target_classes,
                  f"Cross-Domain ({target_name}, fusion)", axes4[1])
    plt.colorbar(im1, ax=axes4[0], fraction=0.045, pad=0.04)
    plt.colorbar(im2, ax=axes4[1], fraction=0.045, pad=0.04)
    fig4.suptitle(f"Confusion Matrices — {source_name} → {target_name}",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig4, "figure4_confusion_matrices.png")

    # Fig 5: top misclassification pairs (target domain, fusion).
    fig5, ax5 = plt.subplots(figsize=(11, 8))
    if len(df_wp) == 0:
        ax5.set_title("No errors")
    else:
        top = df_wp.head(20)
        labels = [f"{r['True_Label'][:24]} → {r['Pred_Label'][:24]}" for _, r in top.iterrows()]
        ax5.barh(range(len(top)), top["Count"].values, color="#e74c3c", alpha=0.8,
                edgecolor="k", lw=0.5)
        ax5.set_yticks(range(len(top))); ax5.set_yticklabels(labels, fontsize=8)
        for i, v in enumerate(top["Count"].values):
            ax5.text(v+1, i, str(v), va="center", fontsize=8)
        ax5.set_xlabel("Misclassified Count", fontsize=10)
        ax5.set_title(f"Top Confusion Pairs — Cross-Domain Fusion ({target_name})",
                      fontweight="bold")
        ax5.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    savefig(fig5, "figure5_misclassification_pairs.png")

    # Fig 6: ECE reliability diagram (in-domain vs cross-domain, both fusion-scale).
    def reliability_diag(probs, yt, title, ax, n_bins=15):
        conf = probs.max(1); pred = source_arr[probs.argmax(1)]
        corr = (pred == np.asarray(yt)).astype(float)
        bins = np.linspace(0, 1, n_bins+1); ba=[]; bc=[]; bn=[]
        for i in range(n_bins):
            m = (conf >= bins[i]) & (conf < bins[i+1])
            ba.append(corr[m].mean() if m.sum()>0 else 0)
            bc.append(conf[m].mean() if m.sum()>0 else (bins[i]+bins[i+1])/2)
            bn.append(int(m.sum()))
        ece = sum(abs(a-c)*n for a,c,n in zip(ba,bc,bn)) / max(sum(bn), 1)
        mid = [(bins[i]+bins[i+1])/2 for i in range(n_bins)]
        ax.plot([0,1],[0,1],"--",color="gray",lw=1.5,label="Perfect calibration")
        ax.bar(mid, ba, width=1/n_bins, alpha=0.7, color="#3498db", edgecolor="k", lw=0.5, label="Accuracy")
        ax.bar(mid, bc, width=1/n_bins, alpha=0.25, color="red", edgecolor="k", lw=0.5, label="Confidence")
        ax.set_title(f"{title}\nECE = {ece*100:.2f}%", fontsize=9, fontweight="bold")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel("Confidence", fontsize=9); ax.set_ylabel("Accuracy", fontsize=9)
    fig6, axes6 = plt.subplots(1, 2, figsize=(10, 5))
    reliability_diag(softmax_np(probe_logit_s, 1.0), s_labels_np,
                     f"In-Domain ({source_name})", axes6[0])
    reliability_diag(fused_eval_calibrated, t_labels_eval,
                     f"Cross-Domain ({target_name}, fusion)", axes6[1])
    fig6.suptitle(f"Reliability Diagrams — {source_name} → {target_name}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig6, "figure6_ece_reliability.png")

    # Fig 7: t-SNE of source vs target embeddings.
    tick("Computing t-SNE embedding...")
    tsne_idx=[]; tsne_lab=[]; tsne_type=[]
    for cls in source_classes:
        idx = stest_meta[stest_meta["class_name"] == cls]["pos_idx"].values
        sel = idx[:min(len(idx), 150)]
        tsne_idx.extend(sel); tsne_lab.extend([cls]*len(sel)); tsne_type.extend(["source"]*len(sel))
    for cls in target_classes:
        idx = ttest_meta[ttest_meta["class_name"] == cls]["pos_idx"].values
        sel = idx[:min(len(idx), 150)]
        tsne_idx.extend(sel); tsne_lab.extend([cls]*len(sel)); tsne_type.extend(["target"]*len(sel))
    tsne_emb = dino_emb[np.array(tsne_idx)].numpy()
    n_tsne = len(tsne_emb)
    perp = min(40, max(5, n_tsne // 3 - 1))
    tsne_out = TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=1000, learning_rate="auto", init="pca").fit_transform(tsne_emb)
    tsne_type_arr = np.array(tsne_type)
    fig7, ax7 = plt.subplots(figsize=(9, 7))
    for split, col, mk in [("source", "#3498db", "o"), ("target", "#e74c3c", "^")]:
        mask = tsne_type_arr == split
        ax7.scatter(tsne_out[mask,0], tsne_out[mask,1], c=col,
                   s=10 if split=="source" else 18, alpha=0.45 if split=="source" else 0.7,
                   marker=mk, label=f"{split.capitalize()} ({mask.sum()} images)")
    ax7.legend(fontsize=10); ax7.axis("off")
    ax7.set_title(f"t-SNE: {source_name} (●) vs {target_name} (▲)\n"
                  f"n={n_tsne}  perplexity={perp}", fontweight="bold")
    plt.tight_layout()
    savefig(fig7, "figure7_tsne.png")


    section(f"[{direction_label}] Results Summary")
    print(f"""
{'='*76}
  {source_name} → {target_name}   (cross-domain, {len(target_classes)} shared classes)
{'='*76}
  In-domain probe ({source_name} test)     : acc={met_in_domain['acc']*100:.2f}%  F1={met_in_domain['f1']*100:.2f}%
  Cross-domain, CLIP only (zero-shot)      : acc={met_cross_clip['acc']*100:.2f}%  F1={met_cross_clip['f1']*100:.2f}%
  Cross-domain, probe only                 : acc={met_cross_probe['acc']*100:.2f}%  F1={met_cross_probe['f1']*100:.2f}%
  Cross-domain, BCVSA fusion (0-shot)      : acc={met_fusion['acc']*100:.2f}%  F1={met_fusion['f1']*100:.2f}%  ECE={ece_fusion*100:.2f}%
  Cross-domain, BCVSA fusion (3-shot)      : acc={met_3shot['acc']*100:.2f}%  F1={met_3shot['f1']*100:.2f}%
  Cross-domain, BCVSA fusion (5-shot)      : acc={met_5shot['acc']*100:.2f}%  F1={met_5shot['f1']*100:.2f}%
  {few_shot_str}

  Fusion parameters: alpha={alpha_f}  T_probe={T_probe_f}  T_clip={T_clip_f}
  Saved figures ({len(saved_figs)}): {', '.join(os.path.basename(f) for f in saved_figs)}
  Results directory: {RESULTS_DIR}
  Runtime: {elapsed()}
{'='*76}
""")

    return {
        "direction": f"{source_name}→{target_name}", "S_n": S_n, "T_n": len(target_classes),
        "in_domain_acc": round(met_in_domain["acc"]*100,2), "in_domain_f1": round(met_in_domain["f1"]*100,2),
        "clip_acc": round(met_cross_clip["acc"]*100,2), "clip_f1": round(met_cross_clip["f1"]*100,2),
        "probe_acc": round(met_cross_probe["acc"]*100,2), "probe_f1": round(met_cross_probe["f1"]*100,2),
        "fusion0_acc": round(met_fusion["acc"]*100,2), "fusion0_f1": round(met_fusion["f1"]*100,2),
        "fusion3_acc": round(met_3shot["acc"]*100,2), "fusion3_f1": round(met_3shot["f1"]*100,2),
        "fusion5_acc": round(met_5shot["acc"]*100,2), "fusion5_f1": round(met_5shot["f1"]*100,2),
        "fusion3_acc_mean": round(float(np.mean(multi_seed_acc3))*100, 2) if RUN_MULTI_SEED else None,
        "fusion3_acc_std":  round(float(np.std(multi_seed_acc3))*100, 2) if RUN_MULTI_SEED else None,
        "fusion5_acc_mean": round(float(np.mean(multi_seed_acc5))*100, 2) if RUN_MULTI_SEED else None,
        "fusion5_acc_std":  round(float(np.std(multi_seed_acc5))*100, 2) if RUN_MULTI_SEED else None,
        "ece_fusion0": round(ece_fusion*100,2),
        "alpha": alpha_f, "T_probe": T_probe_f, "T_clip": T_clip_f,
    }


dir1_path = os.path.join(OUTPUT_DIR, "direction1_PV_to_PD")
dir2_path = os.path.join(OUTPUT_DIR, "direction2_PD_to_PV")

section("Direction 1: PlantVillage → PlantDoc")
res1 = run_direction(dir1_path, "PV to PD", "PlantVillage", "PlantDoc")

section("Direction 2: PlantDoc → PlantVillage")
res2 = run_direction(dir2_path, "PD to PV", "PlantDoc", "PlantVillage")


section("Combined Cross-Dataset Comparison")
df_cmp = pd.DataFrame([res1, res2])
cmp_path = os.path.join(OUTPUT_DIR, "cross_dataset_comparison.csv")
df_cmp.to_csv(cmp_path, index=False)
print(df_cmp.to_string(index=False))

metrics_cb = ["clip_acc", "probe_acc", "fusion0_acc", "fusion3_acc", "fusion5_acc"]
labels_cb  = ["CLIP\nonly", "Probe\nonly", "Fusion\n0-shot", "Fusion\n3-shot", "Fusion\n5-shot"]
v1 = [res1[m] for m in metrics_cb]
v2 = [res2[m] for m in metrics_cb]
x_cb = np.arange(len(labels_cb)); w_cb = 0.35
fig_cb, ax_cb = plt.subplots(figsize=(11, 5.5))
b1 = ax_cb.bar(x_cb - w_cb/2, v1, w_cb, label=res1["direction"], color="#2563EB", alpha=0.85)
b2 = ax_cb.bar(x_cb + w_cb/2, v2, w_cb, label=res2["direction"], color="#DC2626", alpha=0.85)
for bars, vals in [(b1, v1), (b2, v2)]:
    for rect, v in zip(bars, vals):
        ax_cb.text(rect.get_x()+rect.get_width()/2, v+0.5, f"{v:.1f}",
                   ha="center", va="bottom", fontsize=8, fontweight="bold")
ax_cb.set_xticks(x_cb); ax_cb.set_xticklabels(labels_cb, fontsize=10)
ax_cb.set_ylim(0, 108); ax_cb.set_ylabel("Cross-Domain Accuracy (%)", fontsize=12)
ax_cb.set_title("Cross-Dataset Evaluation: PlantVillage ↔ PlantDoc\n"
                "Cross-domain accuracy on a shared class taxonomy, both directions",
                fontsize=12.5, fontweight="bold")
ax_cb.legend(fontsize=10); ax_cb.grid(alpha=0.3, axis="y")
plt.tight_layout()
cmp_fig_path = os.path.join(OUTPUT_DIR, "cross_dataset_comparison_figure.png")
fig_cb.savefig(cmp_fig_path, dpi=180, bbox_inches="tight")
plt.close(fig_cb)
print(f"  Saved: cross_dataset_comparison_figure.png")

section(f"Cross-Dataset Evaluation Complete  ({elapsed()})")
print(f"""
{'='*76}
  CROSS-DATASET EVALUATION COMPLETE  ({elapsed()})
{'='*76}

  Direction 1: PlantVillage → PlantDoc
    In-domain acc={res1['in_domain_acc']:.2f}%  |  Cross-domain fusion 0-shot={res1['fusion0_acc']:.2f}%  "
    5-shot={res1['fusion5_acc']:.2f}%

  Direction 2: PlantDoc → PlantVillage
    In-domain acc={res2['in_domain_acc']:.2f}%  |  Cross-domain fusion 0-shot={res2['fusion0_acc']:.2f}%  "
    5-shot={res2['fusion5_acc']:.2f}%

  Output Files
    {cmp_path}
    {cmp_fig_path}
    direction1_PV_to_PD/results/
    direction2_PD_to_PV/results/
{'='*76}
""")