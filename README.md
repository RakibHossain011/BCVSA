# BCVSA: Bias-Calibrated Visual-Semantic Alignment for Generalized Zero-Shot Plant Disease Recognition

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20thesis-orange)

A generalized zero-shot learning (GZSL) framework for plant disease recognition that combines a **frozen DINOv2-Large** vision backbone with a **frozen CLIP ViT-L/14** vision-language model through a **validation-calibrated fusion mechanism**. The framework is extended with few-shot adaptation, training-free semantic severity estimation, federated learning, a privacy-aware DP-FedAvg-style mechanism, and cross-dataset domain generalization.

The main experiments cover **43 disease classes spanning 15 crops**.

---

## Overview

Most plant disease classifiers work primarily on diseases that were explicitly represented during training. In practical agricultural settings, new crop-disease combinations can appear continuously, while collecting large labeled datasets for every possible class is expensive.

This project investigates a central generalized zero-shot learning question:

> **Can a model recognize a plant disease class for which no task-specific training image was provided?**

BCVSA combines two complementary frozen foundation models:

- **DINOv2-Large** — a self-supervised vision transformer that provides general-purpose visual features and powers a lightweight linear probe trained only on the 26 seen disease classes.
- **CLIP ViT-L/14** — a vision-language model that provides native zero-shot transfer through image-text similarity and supports both seen and unseen disease classes without task-specific training on those unseen classes.

A **bias-calibration module**, tuned only using the designated validation/calibration data, fuses the two branches and reduces the natural seen-class bias of the supervised branch.

The framework is evaluated using the Generalized Zero-Shot Learning (GZSL) protocol with:

- **S** — seen-class accuracy
- **U** — unseen-class accuracy
- **H** — harmonic mean of seen and unseen accuracy

The harmonic mean is the primary GZSL metric because it rewards balanced performance across seen and unseen classes.

---

## What Makes This Different

Zero-shot CLIP for plant disease recognition, federated learning for plant disease classification, and classical embedding-based ZSL methods such as DeViSE, ESZSL, and SAE all have prior work.

The contribution of this project is the **specific combination and evaluation protocol**, including:

- A **calibrated decision-level fusion of a frozen self-supervised vision backbone and a frozen vision-language model**, rather than relying on a single branch or a trainable semantic alignment head.
- Direct comparison against **DeViSE, ESZSL, SAE, CLIP-only, and a trainable alignment-head ablation**.
- **Training-free, segmentation-free semantic severity estimation** using embedding-space healthy/diseased centroids.
- **Federated learning with local-only client baselines**, making the federated improvement interpretable under the same simulated non-IID partition.
- A privacy-aware federated experiment using **client-update clipping and Gaussian perturbation**, with the scope of the privacy claim explicitly limited because no formal `(ε, δ)` accounting is provided.
- A separated **50/50 unseen calibration/evaluation protocol**, multi-seed few-shot evaluation, and calibration-error reporting.

---

## Method

```text
                           Input Image
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       DINOv2-Large (frozen)          CLIP ViT-L/14 (frozen)
                 │                           │
                 ▼                           ▼
        Linear Probe                  Image-text similarity
     (seen classes only)              with text prototypes
                 │                           │
                 │                    10 prompts / class
                 │                    crop + condition
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    Bias-Calibration Module
                               │
        s_c = α·P_probe(c) + (1−α)·P_CLIP(c)
                         − γ·1[c ∈ seen]
                               │
                    α, γ, T tuned on
                    designated calibration data
                               │
                               ▼
                  Prediction over all 43 classes
                               │
                               ▼
                Semantic Severity Estimator
             healthy/diseased embedding centroids
                    (no additional training)
```

### Evaluation protocol

- **26 seen classes** use a standard train/validation/test split.
- **12 unseen classes** are separated into a calibration half and an evaluation half.
- The unseen calibration half is used for calibration-related parameter selection.
- The unseen evaluation half is held out for final evaluation.
- The primary GZSL metric is:

```text
H = 2 × S × U / (S + U)
```

- Few-shot robustness is evaluated across five random seeds: `42, 123, 456, 789, 1000`.
- Temperature search uses a strict-improvement tie-break to avoid selecting a value merely because of search iteration order.

---

## Repository Structure

```text
BCVSA/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── data_preparation.py
│   # Builds the unified 43-class dataset from
│   # PlantVillage + PlantDoc + Cassava
│
├── extract_features.py
│   # Extracts frozen DINOv2-Large embeddings
│
├── main_bcvsa.py
│   # Core pipeline:
│   # probe training, CLIP prototypes, calibration,
│   # GZSL evaluation, few-shot experiments,
│   # ablations, severity estimation, FL, privacy-aware FL,
│   # figures, and result export
│
├── baseline_comparison.py
│   # DeViSE, ESZSL, and SAE GZSL baselines
│
├── xai_backbone_robustness.py
│   # DINOv2 attention visualization and
│   # DINOv2-Small vs DINOv2-Large robustness analysis
│
├── prepare_cross_eval_data.py
│   # Builds the PlantVillage ↔ PlantDoc cross-domain split
│
├── extract_cross_features.py
│   # Feature extraction for cross-domain evaluation
│
├── cross_domain_eval.py
│   # Cross-domain, not cross-class, generalization evaluation
│
└── metadata/
    └── results/
        # Small CSV, JSON, and PNG result artifacts
```

### Files intentionally excluded from Git

The repository does **not** include:

- Raw datasets
- Processed image datasets
- Large embedding files
- PyTorch checkpoints
- Large binary artifacts
- Virtual environments
- Local logs or temporary files

These are excluded through `.gitignore`.

---

## Dataset Access

This repository does **not** redistribute the PlantVillage, PlantDoc, or Cassava Leaf Disease datasets.

Users must obtain the datasets independently from their respective official sources and comply with the individual dataset licenses and terms of use.

Before running the pipeline, configure the local dataset path expected by the scripts.

The dataset license terms are separate from the MIT license covering this repository's code.

---

## Setup

### Requirements

- Python 3.9+
- PyTorch
- Transformers
- NumPy
- SciPy
- scikit-learn
- pandas
- Pillow
- tqdm
- matplotlib

A GPU is recommended for feature extraction and model training.

The pipeline supports CUDA and Intel Arc/XPU environments when the corresponding PyTorch/IPEX configuration is available. The baseline comparison script does not require a GPU.

### Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the environment dependencies used by the project.

For Intel Arc/XPU acceleration, install the compatible Intel Extension for PyTorch configuration for your PyTorch version.

### Dataset path

Update the local `DATA_ROOT` path in the relevant scripts before running the pipeline.

---

## Reproducing Results

Run the main pipeline in the following order:

```bash
# 1. Build the unified dataset
python data_preparation.py

# 2. Extract frozen DINOv2 features
python extract_features.py

# 3. Run the main BCVSA experimental pipeline
python main_bcvsa.py

# 4. Run classical GZSL baselines
python baseline_comparison.py
```

Optional analyses:

```bash
# XAI attention visualization + backbone robustness
python xai_backbone_robustness.py

# Cross-domain dataset preparation
python prepare_cross_eval_data.py

# Cross-domain feature extraction
python extract_cross_features.py

# PlantVillage ↔ PlantDoc domain generalization evaluation
python cross_domain_eval.py
```

Results are written to:

```text
metadata/results/
```

in CSV, JSON, and PNG formats.

The main experiments use fixed seeds for reproducibility. Few-shot and federated experiments additionally use multiple seeds where reported.

---

## Technical Components

- **Deep learning with PyTorch** — custom training loops, mixed precision, and hardware-aware optimization.
- **Frozen foundation models** — DINOv2 self-supervised visual features and CLIP vision-language features.
- **Generalized Zero-Shot Learning** — calibrated stacking, seen/unseen bias correction, and harmonic-mean evaluation.
- **Few-shot learning** — prototype adaptation with multi-seed robustness analysis.
- **Federated learning** — FedAvg across simulated non-IID clients with local-only comparisons.
- **Privacy-aware federated learning** — client-update clipping and Gaussian perturbation.
- **Model calibration** — temperature scaling, Expected Calibration Error, and reliability analysis.
- **Explainable AI** — attention-map visualization with artifact-aware post-processing.
- **Experimental methodology** — train/validation/test separation, unseen calibration/evaluation separation, ablation studies, and multi-seed reporting.
- **Data engineering** — multi-source dataset unification, deduplication, and class organization.
- **Scientific communication** — result export and publication-oriented figure generation.

---

## Limitations & Honest Notes

Being explicit about limitations is an important part of the experimental design.

- **Zero-shot does not mean zero prior information.** CLIP was pretrained on web-scale image-text data and may have encountered plant or leaf imagery before this task. Here, zero-shot means that the model receives no task-specific fine-tuning images for the unseen disease classes.
- **Calibration uses labeled unseen-class samples.** The designated unseen calibration split is used for calibration-related parameter selection. Therefore, the protocol should not be described as purely inductive zero-shot evaluation.
- **Unseen calibration remains weaker than seen calibration.** BCVSA obtains ECE ≈ 1.5% on the seen branch but ≈ 30% on the unseen branch.
- **Per-class performance is heterogeneous.** Visually similar healthy-leaf morphology and closely related crop/disease appearances can remain challenging for zero-shot recognition.
- **Federated clients are simulated.** The Bangladesh, India, USA, and Spain labels represent simulated crop-based non-IID client partitions, not literal geographic data collection.
- **Privacy mechanism is not a formal DP guarantee.** The federated privacy experiment uses update clipping and Gaussian perturbation, but no certified `(ε, δ)` privacy budget has been computed. A formal privacy accountant would be required before claiming a specific differential privacy guarantee.
- **Semantic severity is not clinical severity.** The severity experiment measures embedding-space separation between healthy and diseased semantic centroids. It is not a clinically validated plant disease severity score.
- **Cross-domain evaluation is domain generalization, not cross-class zero-shot evaluation.** PlantVillage and PlantDoc experiments should be interpreted as domain-transfer analysis.

---

## References & Related Work

- Xian, Y., Lampert, C. H., Schiele, B., & Akata, Z. (2018). *Zero-Shot Learning — A Comprehensive Evaluation of the Good, the Bad and the Ugly.* IEEE TPAMI.
- Chao, W.-L., Changpinyo, S., Gong, B., & Sha, F. (2016). *An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild.* ECCV.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- Frome, A. et al. (2013). *DeViSE: A Deep Visual-Semantic Embedding Model.* NeurIPS.
- Romera-Paredes, B., & Torr, P. (2015). *An Embarrassingly Simple Approach to Zero-Shot Learning.* ICML.
- Kodirov, E., Xiang, T., & Gong, S. (2017). *Semantic Autoencoder for Zero-Shot Learning.* CVPR.
- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
- Darcet, T. et al. (2023). *Vision Transformers Need Registers.*

### Datasets

The project uses PlantVillage, PlantDoc, and Cassava Leaf Disease data. Each dataset is governed by its own license and usage terms.

---

## License

This repository's source code is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

**Important:** the MIT license applies to the code in this repository. It does **not** grant rights to redistribute the PlantVillage, PlantDoc, or Cassava datasets. Their respective licenses remain applicable.

---

## Acknowledgments

This project builds on:

- DINOv2 from Meta AI
- CLIP from OpenAI
- Hugging Face Transformers
- PyTorch and the broader open-source machine learning ecosystem

---

## Author

**Mohammad Rakib Hossain**

- GitHub: [RakibHossain011](https://github.com/RakibHossain011)
- LinkedIn: [Mohammad Rakib Hossain](https://www.linkedin.com/in/mohammad-rakib-hossain/)

This project was developed as part of thesis research at the University of Dhaka (NITER).

Feedback, questions, and research collaboration are welcome.
