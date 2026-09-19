# BCVSA: Bias-Calibrated Visual-Semantic Alignment for Generalized Zero-Shot Plant Disease Recognition

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20thesis-orange)

## What This Is

BCVSA is a generalized zero-shot learning (GZSL) framework for plant disease recognition — a system built to correctly identify plant disease classes it was **never explicitly trained on**, spanning 43 disease classes across 15 crops. It's designed as a single end-to-end pipeline rather than a standalone classifier, combining self-supervised computer vision, vision-language models, few-shot adaptation, federated learning, and privacy-aware training.

## Why This Project

- New crop-disease combinations keep appearing faster than labeled datasets can realistically be built, and most classifiers simply fail on anything outside their training set.
- Agricultural data is often siloed across farms, regions, and institutions that can't or won't share raw images, limiting how much a shared model can improve.
- Existing plant disease AI work is largely single-purpose — strong at classification, but rarely tackling unseen diseases, scarce labels, decentralized data, and prediction trustworthiness together in one system.

This project is built directly against those constraints, rather than around a single benchmark number.

## Why It's Good

- **One coherent system, not disconnected experiments** — zero-shot recognition, few-shot adaptation, disease severity estimation, federated training, and privacy-aware learning all live inside one reproducible pipeline.
- **Designed for deployment realities, not just leaderboard accuracy** — limited labels, distributed data ownership, and trustworthy/calibrated predictions are treated as core design constraints, not afterthoughts.
- **Held to research-grade rigor** — careful separation of training, validation, and unseen data, multi-seed robustness checks, and confidence calibration are built into the evaluation from the ground up.

## Tech & Skills Used

**Data Engineering**
- Multi-source dataset unification (PlantVillage, PlantDoc, Cassava Leaf Disease)
- Data cleaning, deduplication, and cross-source class harmonization
- Reproducible, cached feature-extraction pipelines

**Machine Learning & Deep Learning**
- Frozen foundation models (DINOv2, CLIP) with lightweight trainable heads
- Generalized zero-shot and few-shot learning
- Model calibration and reliability analysis

**Applied AI Systems**
- Federated learning across simulated distributed clients
- Privacy-aware training (update clipping + noise perturbation)
- Explainable AI via attention-based visualization

**Engineering**
- PyTorch, Hugging Face Transformers
- CUDA / Intel Arc (XPU) hardware support
- Modular pipeline: data → features → training → evaluation → reporting

---

**Mohammad Rakib Hossain**
[GitHub](https://github.com/RakibHossain011) · [LinkedIn](https://www.linkedin.com/in/mohammad-rakib-hossain/)
Thesis research, University of Dhaka (NITER)
