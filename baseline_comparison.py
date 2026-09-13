"""
baseline_comparison.py

"""

import os
import sys
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import solve_sylvester
from sklearn.metrics import f1_score, precision_score, recall_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

DATA_ROOT = r"E:\VS Code\Projects\Thesis\TP002\PlantF\Data"

SAVE_DIR        = os.path.join(DATA_ROOT, "metadata")
BUNDLE_PATH     = os.path.join(SAVE_DIR, "PLANT_bundle.pkl")
TEXT_CACHE_PATH = os.path.join(SAVE_DIR, "clip_text_prototypes.pt")
RESULTS_CSV     = os.path.join(SAVE_DIR, "results", "results_summary.csv")

TRAIN_DIR       = os.path.join(DATA_ROOT, "global", "train")
VAL_DIR         = os.path.join(DATA_ROOT, "global", "val")
SEEN_TEST_DIR   = os.path.join(DATA_ROOT, "global", "test")
UNSEEN_TEST_DIR = os.path.join(DATA_ROOT, "zero_shot", "unseen_test")

OUT_DIR = os.path.join(SAVE_DIR, "results", "baselines")

GLOBAL_SEED       = 42
UNSEEN_SPLIT_SEED = 42

DEVICE = torch.device("cpu")

torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(str(msg))


def section(msg):
    log("\n" + "=" * 78)
    log(msg)
    log("=" * 78)


# =============================================================================
# Metric helpers
# =============================================================================

def pacc(pred, true):
    return float((np.asarray(pred) == np.asarray(true)).mean())


def hm(s, u):
    return 2 * s * u / (s + u + 1e-9)


def full_metrics(y_true, y_pred, labels):
    return {
        "acc":  pacc(y_pred, y_true),
        "f1":   f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "prec": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "rec":  recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
    }


def softmax_np(x, T=1.0):
    x = np.asarray(x, dtype=np.float64) / T
    e = np.exp(x - x.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def l2norm_rows(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


# =============================================================================
# Load data
# =============================================================================

def require_file(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nMissing required file for {what}:\n  {path}\n\n"
            f"Run extract_features.py / main_bcvsa.py first."
        )


def load_data():
    section("[1] Loading cached features and metadata")
    require_file(BUNDLE_PATH, "DINOv2 visual embeddings")
    require_file(TEXT_CACHE_PATH, "CLIP text (semantic) prototypes")
    for d in [TRAIN_DIR, VAL_DIR, SEEN_TEST_DIR, UNSEEN_TEST_DIR]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing expected directory: {d}")

    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)

    meta = pd.DataFrame(bundle["metadata"])
    meta.index = range(len(meta))
    meta["pos_idx"] = meta.index
    if "class_name" not in meta.columns and "class" in meta.columns:
        meta = meta.rename(columns={"class": "class_name"})
    if "type" not in meta.columns:
        meta["type"] = "seen"
        meta.loc[meta["split"].str.contains("unseen", case=False, na=False), "type"] = "unseen"

    dino_emb = torch.tensor(bundle["embeddings"], dtype=torch.float32)
    dino_emb = torch.nn.functional.normalize(dino_emb, dim=-1).numpy()
    log(f"  Visual embeddings: {dino_emb.shape}")

    seen_classes = sorted([d for d in os.listdir(TRAIN_DIR)
                            if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    unseen_classes = sorted([d for d in os.listdir(UNSEEN_TEST_DIR)
                              if os.path.isdir(os.path.join(UNSEEN_TEST_DIR, d))])
    all_classes = sorted(list(set(seen_classes) | set(unseen_classes)))
    cls2id = {c: i for i, c in enumerate(all_classes)}
    seen_ids = np.array([cls2id[c] for c in seen_classes])
    seen_local = {c: i for i, c in enumerate(seen_classes)}
    all_cls_arr = np.array(all_classes)
    S_n, C = len(seen_classes), len(all_classes)
    log(f"  Seen={S_n}  Unseen={len(unseen_classes)}  All={C}")

    train_meta = meta[(meta["type"] == "seen") & (meta["split"] == "train")].copy()
    val_meta   = meta[(meta["type"] == "seen") & (meta["split"] == "val")].copy()
    stest_meta = meta[(meta["type"] == "seen") & (meta["split"] == "test")].copy()
    utest_meta = meta[meta["type"] == "unseen"].copy()
    train_meta["seen_id"] = train_meta["class_name"].map(seen_local)
    val_meta["seen_id"] = val_meta["class_name"].map(seen_local)

    np.random.seed(UNSEEN_SPLIT_SEED)
    u_meta_arr = np.arange(len(utest_meta))
    u_cls_arr = np.array(utest_meta["class_name"].tolist())
    uval_mask = np.zeros(len(utest_meta), dtype=bool)
    utest_mask = np.zeros(len(utest_meta), dtype=bool)
    for cls in unseen_classes:
        pos = u_meta_arr[u_cls_arr == cls]
        perm = np.random.permutation(len(pos))
        half = len(pos) // 2
        uval_mask[pos[perm[:half]]] = True
        utest_mask[pos[perm[half:]]] = True
    np.random.seed(GLOBAL_SEED)

    log(f"  Unseen calibration half: {int(uval_mask.sum())}   "
        f"evaluation half: {int(utest_mask.sum())}")

    all_text = torch.load(TEXT_CACHE_PATH, weights_only=False)
    all_text = torch.nn.functional.normalize(all_text.float(), dim=-1).numpy()
    if all_text.shape[0] != C:
        raise ValueError(
            f"clip_text_prototypes.pt has {all_text.shape[0]} rows but "
            f"{C} classes were found on disk. Re-run main_bcvsa.py Section 3."
        )
    log(f"  Semantic prototypes: {all_text.shape}")

    return dict(
        dino_emb=dino_emb, all_text=all_text,
        seen_classes=seen_classes, unseen_classes=unseen_classes,
        all_classes=all_classes, all_cls_arr=all_cls_arr,
        seen_ids=seen_ids, S_n=S_n, C=C,
        train_meta=train_meta, val_meta=val_meta,
        stest_meta=stest_meta, utest_meta=utest_meta,
        uval_mask=uval_mask, utest_mask=utest_mask,
    )


# =============================================================================
# Calibrated-stacking GZSL evaluation
# =============================================================================

def adaptive_temperatures(raw_scores, n=10):
    spread = float(np.std(raw_scores))
    if not np.isfinite(spread) or spread < 1e-8:
        spread = 1.0
    mults = np.geomspace(0.01, 4.0, n)
    temps = sorted(set(float(spread * m) for m in mults), reverse=True)
    return temps


def calibrated_gzsl_eval(scores_val_seen, scores_val_unseen_cal,
                          scores_test_seen, scores_test_unseen_eval,
                          seen_ids, all_cls_arr,
                          y_val_seen, y_val_unseen_cal,
                          y_test_seen, y_test_unseen_eval, label=""):
    temps = adaptive_temperatures(
        np.concatenate([scores_val_seen.ravel(), scores_val_unseen_cal.ravel()]))
    coarse_g = np.linspace(-3.0, 3.0, 121, dtype=np.float64)

    best = {"H": -1.0, "T": temps[0], "gamma": 0.0}
    for T in temps:
        sm_v = softmax_np(scores_val_seen, T)
        sm_u = softmax_np(scores_val_unseen_cal, T)
        bH, bg = -1.0, 0.0
        for g in coarse_g:
            sv = sm_v.copy(); su = sm_u.copy()
            sv[:, seen_ids] -= g
            su[:, seen_ids] -= g
            S = pacc(all_cls_arr[sv.argmax(1)], y_val_seen)
            U = pacc(all_cls_arr[su.argmax(1)], y_val_unseen_cal)
            H = hm(S, U)
            if H > bH:
                bH, bg = H, g
        for g in np.linspace(bg - 0.3, bg + 0.3, 61, dtype=np.float64):
            sv = sm_v.copy(); su = sm_u.copy()
            sv[:, seen_ids] -= g
            su[:, seen_ids] -= g
            S = pacc(all_cls_arr[sv.argmax(1)], y_val_seen)
            U = pacc(all_cls_arr[su.argmax(1)], y_val_unseen_cal)
            H = hm(S, U)
            if H > best["H"]:
                best.update({"H": H, "T": T, "gamma": float(g)})

    if best["T"] == min(temps):
        log(f"    [{label}] WARNING: chosen T={best['T']:.4g} is the smallest "
            f"value in the adaptive temperature grid -- saturation ceiling "
            f"not reached. Extend np.geomspace(0.01, 4.0, n) downward "
            f"(e.g. np.geomspace(0.001, 4.0, n)) and re-run.")

    T, g = best["T"], best["gamma"]
    sm_ts = softmax_np(scores_test_seen, T)
    sm_tu = softmax_np(scores_test_unseen_eval, T)
    sm_ts[:, seen_ids] -= g
    sm_tu[:, seen_ids] -= g
    pred_s = all_cls_arr[sm_ts.argmax(1)]
    pred_u = all_cls_arr[sm_tu.argmax(1)]
    S = pacc(pred_s, y_test_seen)
    U = pacc(pred_u, y_test_unseen_eval)
    H = hm(S, U)
    return {"T": T, "gamma": g, "S": S, "U": U, "H": H,
            "pred_s": pred_s, "pred_u": pred_u, "val_H": best["H"]}


def report_row(name, ev, y_test_seen, y_test_unseen_eval, seen_classes, unseen_classes):
    met_s = full_metrics(y_test_seen, ev["pred_s"], seen_classes)
    met_u = full_metrics(y_test_unseen_eval, ev["pred_u"], unseen_classes)
    log(f"  {name:<28} S={ev['S']*100:6.2f}%  U={ev['U']*100:6.2f}%  "
        f"H={ev['H']*100:6.2f}%   (val H during tuning = {ev['val_H']*100:.2f}%, "
        f"T={ev['T']:.4g}, gamma={ev['gamma']:.4f})")
    return {
        "Method": name, "S%": round(ev["S"] * 100, 2), "U%": round(ev["U"] * 100, 2),
        "H%": round(ev["H"] * 100, 2), "F1_Seen%": round(met_s["f1"] * 100, 2),
        "F1_Unseen%": round(met_u["f1"] * 100, 2),
        "T_clip_or_temp": round(float(ev["T"]), 6), "gamma": round(float(ev["gamma"]), 4),
        "Val_H_during_tuning%": round(ev["val_H"] * 100, 2),
    }


# =============================================================================
# Baselines
# =============================================================================

def fit_devise(X_train, y_train_ids, S_seen, S_all, d_out, X_val=None, y_val_ids=None,
               epochs=250, lr=5e-3, margin=1.0, weight_decay=1e-4, patience=40):
    """DeViSE (Frome et al., 2013): linear map, max-margin ranking loss."""
    torch.manual_seed(GLOBAL_SEED)
    n, d_in = X_train.shape

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train_ids, dtype=torch.long)
    S_seen_t = torch.tensor(S_seen, dtype=torch.float32)

    if X_val is not None and y_val_ids is not None:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val_ids, dtype=torch.long)
    else:
        Xv, yv = X, y

    W = nn.Linear(d_in, d_out, bias=False)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_proxy = -1.0
    best_state = None
    bad_epochs = 0
    batch_size = 256

    for epoch in range(epochs):
        W.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], y[idx]
            proj = torch.nn.functional.normalize(W(xb), dim=-1)
            all_scores = proj @ S_seen_t.T
            pos_scores = all_scores.gather(1, yb.view(-1, 1))
            margins = torch.clamp(margin - pos_scores + all_scores, min=0.0)
            margins.scatter_(1, yb.view(-1, 1), 0.0)
            loss = margins.sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W.parameters(), max_norm=5.0)
            opt.step()
            total_loss += float(loss) * len(idx)
        sched.step()

        W.eval()
        with torch.no_grad():
            proj_v = torch.nn.functional.normalize(W(Xv), dim=-1)
            proxy = float((proj_v @ S_seen_t.T).argmax(1).eq(yv).float().mean())
        if proxy > best_proxy:
            best_proxy = proxy
            best_state = {k: v.clone() for k, v in W.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            log(f"    DeViSE epoch {epoch+1:3d}/{epochs}  loss={total_loss/n:.4f}  "
                f"seen-retrieval-proxy={proxy*100:.2f}%  best={best_proxy*100:.2f}%")
        if bad_epochs >= patience:
            log(f"    DeViSE early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        W.load_state_dict(best_state)
    W.eval()
    log(f"    DeViSE final: best seen-retrieval proxy = {best_proxy*100:.2f}%")

    def score_fn(X_query):
        with torch.no_grad():
            proj = torch.nn.functional.normalize(
                W(torch.tensor(X_query, dtype=torch.float32)), dim=-1).numpy()
        return proj @ S_all.T

    return score_fn


def fit_eszsl(X_train, y_train_onehot, S_seen, S_all, gamma_reg=1.0, lambda_reg=1.0):
    """ESZSL (Romera-Paredes & Torr, 2015): closed-form bilinear compatibility, bipolar targets."""
    X = X_train.astype(np.float64)
    Y = (2.0 * y_train_onehot - 1.0).astype(np.float64)
    S = S_seen.astype(np.float64)

    d_x = X.shape[1]
    d_a = S.shape[1]
    A = X.T @ X + gamma_reg * np.eye(d_x)
    B = S.T @ S + lambda_reg * np.eye(d_a)
    M = X.T @ Y @ S
    V0 = np.linalg.solve(A, M)
    V = V0 @ np.linalg.inv(B)

    def score_fn(X_query):
        return X_query.astype(np.float64) @ V @ S_all.T

    return score_fn


def fit_sae(X_train, y_train_ids, S_seen, S_all, lam=1.0):
    """SAE (Kodirov et al., 2017): Sylvester-equation closed-form projection."""
    S_per_sample = S_seen[y_train_ids]

    X = X_train.astype(np.float64).T
    Sm = S_per_sample.astype(np.float64).T

    A = Sm @ Sm.T
    B = lam * (X @ X.T)
    Cmat = (1.0 + lam) * (Sm @ X.T)

    A = A + 1e-6 * np.eye(A.shape[0])
    B = B + 1e-6 * np.eye(B.shape[0])

    W = solve_sylvester(A, B, Cmat)

    def score_fn(X_query):
        proj = X_query.astype(np.float64) @ W.T
        proj = l2norm_rows(proj)
        return proj @ l2norm_rows(S_all.astype(np.float64)).T

    return score_fn


# =============================================================================
# Driver
# =============================================================================

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    d = load_data()
    dino_emb, all_text = d["dino_emb"], d["all_text"]
    seen_classes, unseen_classes = d["seen_classes"], d["unseen_classes"]
    all_cls_arr, seen_ids, S_n = d["all_cls_arr"], d["seen_ids"], d["S_n"]
    train_meta, val_meta, stest_meta, utest_meta = d["train_meta"], d["val_meta"], d["stest_meta"], d["utest_meta"]
    uval_mask, utest_mask = d["uval_mask"], d["utest_mask"]

    S_seen = all_text[seen_ids]

    X_train = dino_emb[train_meta["pos_idx"].values]
    y_train_ids = train_meta["seen_id"].values.astype(int)
    y_train_onehot = np.eye(S_n)[y_train_ids]

    X_val = dino_emb[val_meta["pos_idx"].values]
    y_val = val_meta["class_name"].values
    y_val_ids = val_meta["seen_id"].values.astype(int)

    X_stest = dino_emb[stest_meta["pos_idx"].values]
    y_stest = stest_meta["class_name"].values

    u_pos = utest_meta["pos_idx"].values
    u_cls = utest_meta["class_name"].values
    X_uval = dino_emb[u_pos[uval_mask]]
    y_uval = u_cls[uval_mask]
    X_ueval = dino_emb[u_pos[utest_mask]]
    y_ueval = u_cls[utest_mask]

    log(f"\n  Train (seen): {X_train.shape}   Val (seen): {X_val.shape}   "
        f"Test (seen): {X_stest.shape}")
    log(f"  Unseen calibration: {X_uval.shape}   Unseen evaluation: {X_ueval.shape}")

    rows = []

    section("[2] DeViSE (Frome et al., 2013)")
    devise_score = fit_devise(X_train, y_train_ids, S_seen, all_text, d_out=all_text.shape[1],
                               X_val=X_val, y_val_ids=y_val_ids)
    ev = calibrated_gzsl_eval(
        devise_score(X_val), devise_score(X_uval),
        devise_score(X_stest), devise_score(X_ueval),
        seen_ids, all_cls_arr, y_val, y_uval, y_stest, y_ueval, label="DeViSE")
    rows.append(report_row("DeViSE (2013)", ev, y_stest, y_ueval, seen_classes, unseen_classes))

    section("[3] ESZSL (Romera-Paredes & Torr, 2015)")
    best_eszsl = {"H": -1.0}
    for gamma_reg in [0.1, 1.0, 10.0, 100.0]:
        for lambda_reg in [0.1, 1.0, 10.0, 100.0]:
            try:
                score_fn = fit_eszsl(X_train, y_train_onehot, S_seen, all_text,
                                      gamma_reg=gamma_reg, lambda_reg=lambda_reg)
                sv, su = score_fn(X_val), score_fn(X_uval)
                if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(su))):
                    raise np.linalg.LinAlgError("non-finite scores")
                ev_try = calibrated_gzsl_eval(
                    sv, su, sv, su, seen_ids, all_cls_arr, y_val, y_uval, y_val, y_uval,
                    label=f"ESZSL g={gamma_reg} l={lambda_reg}")
            except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                log(f"    skipping gamma_reg={gamma_reg} lambda_reg={lambda_reg}: {e}")
                continue
            if ev_try["H"] > best_eszsl["H"]:
                best_eszsl = {"H": ev_try["H"], "gamma_reg": gamma_reg,
                              "lambda_reg": lambda_reg, "score_fn": score_fn}
    if best_eszsl["H"] < 0:
        raise RuntimeError("ESZSL: every (gamma_reg, lambda_reg) combination failed.")
    log(f"  Selected on validation only: gamma_reg={best_eszsl['gamma_reg']}  "
        f"lambda_reg={best_eszsl['lambda_reg']}  (val H={best_eszsl['H']*100:.2f}%)")
    eszsl_score = best_eszsl["score_fn"]
    ev = calibrated_gzsl_eval(
        eszsl_score(X_val), eszsl_score(X_uval),
        eszsl_score(X_stest), eszsl_score(X_ueval),
        seen_ids, all_cls_arr, y_val, y_uval, y_stest, y_ueval, label="ESZSL final")
    rows.append(report_row("ESZSL (2015)", ev, y_stest, y_ueval, seen_classes, unseen_classes))

    section("[4] SAE -- Semantic AutoEncoder (Kodirov et al., 2017)")
    best_sae = {"H": -1.0}
    for lam in [0.01, 0.1, 1.0, 10.0, 100.0]:
        try:
            score_fn = fit_sae(X_train, y_train_ids, S_seen, all_text, lam=lam)
            sv, su = score_fn(X_val), score_fn(X_uval)
            if not (np.all(np.isfinite(sv)) and np.all(np.isfinite(su))):
                raise np.linalg.LinAlgError("non-finite scores")
            ev_try = calibrated_gzsl_eval(
                sv, su, sv, su, seen_ids, all_cls_arr, y_val, y_uval, y_val, y_uval,
                label=f"SAE lambda={lam}")
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            log(f"    skipping lambda={lam}: {e}")
            continue
        if ev_try["H"] > best_sae["H"]:
            best_sae = {"H": ev_try["H"], "lam": lam, "score_fn": score_fn}
    if best_sae["H"] < 0:
        raise RuntimeError("SAE: every lambda failed.")
    log(f"  Selected on validation only: lambda={best_sae['lam']}  "
        f"(val H={best_sae['H']*100:.2f}%)")
    sae_score = best_sae["score_fn"]
    ev = calibrated_gzsl_eval(
        sae_score(X_val), sae_score(X_uval),
        sae_score(X_stest), sae_score(X_ueval),
        seen_ids, all_cls_arr, y_val, y_uval, y_stest, y_ueval, label="SAE final")
    rows.append(report_row("SAE (2017)", ev, y_stest, y_ueval, seen_classes, unseen_classes))

    section("[5] Merging with your existing results (read-only)")
    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        keep_cols = ["Method", "S%", "U%", "H%", "F1_Seen%", "F1_Unseen%"]
        for _, r in df_existing.iterrows():
            row = {c: r[c] if c in df_existing.columns else "-" for c in keep_cols}
            row["T_clip_or_temp"] = "-"
            row["gamma"] = "-"
            row["Val_H_during_tuning%"] = "-"
            rows.append(row)
        log(f"  Merged {len(df_existing)} existing rows from {RESULTS_CSV}")
    else:
        log(f"  {RESULTS_CSV} not found -- reporting the three new baselines only.")

    section("[6] Saving results")
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "baseline_comparison.csv")
    df_out.to_csv(csv_path, index=False)
    log(f"  Saved: {csv_path}")

    json_path = os.path.join(OUT_DIR, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump({
            "protocol": "Same seen/unseen split, same 50/50 unseen "
                        "calibration/evaluation split (seed=42), same "
                        "calibrated-stacking bias correction as main_bcvsa.py.",
            "fl_run": False,
            "results": rows,
        }, f, indent=2)
    log(f"  Saved: {json_path}")

    plot_rows = [r for r in rows if isinstance(r.get("H%"), (int, float))]
    if plot_rows:
        names = [r["Method"] for r in plot_rows]
        S_vals = [r["S%"] for r in plot_rows]
        U_vals = [r["U%"] for r in plot_rows]
        H_vals = [r["H%"] for r in plot_rows]
        x = np.arange(len(names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5.5))
        ax.bar(x - w, S_vals, w, label="Seen S%", color="#2563EB")
        ax.bar(x,     U_vals, w, label="Unseen U%", color="#DC2626")
        ax.bar(x + w, H_vals, w, label="Harmonic Mean H%", color="#16A34A")
        for xi, vals in zip(x, zip(S_vals, U_vals, H_vals)):
            for off, v in zip([-w, 0, w], vals):
                ax.text(xi + off, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.set_title("GZSL Baseline Comparison", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "baseline_comparison_figure.png")
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: {fig_path}")

    log_path = os.path.join(OUT_DIR, "baseline_comparison_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"  Saved: {log_path}")

    section("DONE")
    log(df_out.to_string(index=False))
    log(f"\nTotal runtime: {(time.time()-t0)/60:.2f} min")
    log(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 78)
        print("baseline_comparison.py stopped with an error:")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 78)
        sys.exit(1)