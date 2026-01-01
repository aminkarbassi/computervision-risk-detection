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

The system is deliberately split into **independent components**, following industry best practices:

### Training Pipeline
- PyTorch-based image classification
- Config-driven training
- Experiment tracking with MLflow
- Model versioning via MLflow Model Registry
- Data lineage via deterministic dataset hashing
- Fully containerized with Docker

### Inference Pipeline
- Lightweight, CPU-based inference
- Separate Docker image from training
- Stateless prediction on individual images
- Designed to extend to batch and API-based serving

---

## Key Engineering Features

### MLOps & Reproducibility
- Experiment tracking (parameters, metrics, artifacts)
- Model versioning and lifecycle management
- Explicit linkage between:
  - code version (git commit)
  - data version (hash-based fingerprint)
  - trained model artifact

### Production-Oriented Design
- Clear separation of training vs inference
- No data baked into Docker images
- Deterministic, repeatable runs
- Clean project structure suitable for CI/CD extension

---

## Running the System (High-Level)

### Training (Docker)
A dedicated Docker image is used for training. Data and outputs are mounted at runtime to ensure reproducibility and portability.

### Inference (Docker)
A separate inference image loads a trained model and runs predictions on provided images, without requiring training data.

> Detailed commands are intentionally kept minim

