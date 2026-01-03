# CV Risk Detection – Production-Grade Computer Vision & MLOps

This repository contains a production-oriented computer vision system for **risk indicator detection from aerial imagery**, designed to mirror real-world ML engineering practices used in insurance and geospatial analytics.

The project focuses on **end-to-end ML system design**, including reproducible training, containerized inference, experiment tracking, model versioning, and data lineage — not just model accuracy.

---

## Problem Statement

Insurance risk assessment increasingly relies on **aerial and satellite imagery** to identify risk indicators such as:
- roof presence and condition
- solar panel installations
- surrounding land use context

This project demonstrates how to build a **scalable, auditable ML pipeline** that can support such use cases in production environments.

---

## System Overview

The system is deliberately split into **independent components**, following industry best practices.

### Training Pipeline
- PyTorch-based model training
- Config-driven experiments (YAML)
- Experiment tracking with MLflow
- Model artifact management
- Data lineage via deterministic dataset hashing
- Fully containerized with Docker

### Inference Pipeline
- Lightweight, CPU-based inference
- Separate Docker image from training
- Stateless prediction on individual images
- Designed to extend to batch or API-based serving

---

## Key Engineering Features

### MLOps & Reproducibility
- Experiment tracking (parameters, metrics, artifacts)
- Explicit linkage between:
  - code version (git commit)
  - data version (hash-based fingerprint)
  - trained model artifact
- Clear separation of configuration and logic

### Production-Oriented Design
- Training and inference decoupled
- No datasets or artifacts baked into Docker images
- Deterministic, repeatable runs
- Project structure suitable for CI/CD extension

---

## Object Detection on Aerial Imagery (Week 4)

The project was extended from image classification to **object detection on high-resolution aerial imagery**, reflecting real-world geospatial computer vision constraints.

### Key additions
- Conversion of segmentation masks into detection annotations
- Tiling of large aerial images (e.g. 5000×5000) into trainable patches (512×512)
- Bounding box remapping per tile
- PyTorch-compatible detection dataset
- Fine-tuning a Faster R-CNN model with a custom ROI head

### Engineering considerations
- Full-resolution aerial imagery is never trained directly
- Tiling is required to control memory usage and object density
- Pretrained detection models require explicit replacement of ROI heads
- Smoke-test training is used before running full training loops

These steps mirror production geospatial CV pipelines used in industry.

---

## Running the System (High-Level)

### Training
Training is executed inside a Docker container.  
Data and output directories are mounted at runtime to ensure portability and reproducibility.

### Inference
Inference runs in a separate container using a trained model artifact, without requiring access to training data.

> Low-level commands are intentionally omitted; this repository prioritizes architecture and engineering practices over tutorial-style execution.

---

## Current Scope

- Binary image classification
- Object detection on aerial imagery
- CPU-based training and inference
- Small, curated datasets for pipeline validation

---

## Roadmap

Planned extensions include:
- Detection of additional risk indicators (e.g. solar panels)
- Multi-class and multi-modal models
- Detection evaluation metrics (IoU, precision/recall)
- Model serving via API
- Automated retraining and monitoring workflows

---

## Notes

- Open datasets are used for development; proprietary data is intentionally excluded.
- The goal of this project is **engineering credibility**, not benchmark performance.
- Design choices emphasize traceability, extensibility, and correctness.

---
## Object Detection on Aerial Imagery

The project was extended from image classification to **object detection** on high-resolution aerial imagery.

### Key additions
- Conversion of segmentation masks to detection annotations
- Tiling of large aerial images (5000×5000) into trainable patches (512×512)
- Bounding box remapping per tile
- PyTorch-compatible detection dataset
- Faster R-CNN fine-tuning with a custom detection head

### Engineering considerations
- Full-resolution aerial imagery is never trained directly
- Tiling is required to control memory usage and object density
- Pretrained detection models require explicit replacement of ROI heads
- A smoke-test training step is used before full training runs

This mirrors real-world geospatial computer vision pipelines used in production systems.


---

## Disclaimer

This project is for educational and demonstration purposes only and is not affiliated with or representative of any specific company or proprietary system.


