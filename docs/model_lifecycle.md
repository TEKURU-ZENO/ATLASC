# ATLAS — Model Lifecycle Documentation

This document defines the complete lifecycle of machine learning models used in ATLAS,
from data ingestion to training, serving, monitoring, and iteration.

ATLAS strictly separates:
- Offline model training (research phase)
- Online model serving & simulation (product phase)

This separation ensures reproducibility, stability, and production readiness.

---

## 1. Lifecycle Overview

The ATLAS model lifecycle consists of five stages:

1. Data Preparation
2. Model Training
3. Model Validation & Freezing
4. Model Serving & Simulation
5. Iteration & Improvement

Each stage has a clear responsibility and boundary.

---

## 2. Stage 1 — Data Preparation

### Purpose
Convert raw biological data into structured, reproducible inputs for model training.

### Inputs
- scRNA-seq datasets
- ATAC-seq datasets
- Gene regulatory & pathway databases

### Key Steps
- Quality control (filter low-quality cells/genes)
- Normalization & batch correction
- Feature selection / dimensionality reduction
- Graph construction (gene regulatory networks)

### Outputs
- Processed AnnData objects
- Cached embeddings
- Gene/pathway graphs

### Relevant Code
omics/
graphs/
data/processed/
scripts/preprocess_data.py


---

## 3. Stage 2 — Model Training (Offline)

### Purpose
Learn causal, probabilistic representations of cell state transitions under interventions.

### Characteristics
- Offline
- Compute-intensive
- Rarely executed
- Fully configuration-driven

### How Training Is Triggered
```bash
python experiments/train.py --config configs/model.yaml
What Happens
Encoders learn multimodal cell representations

GNN learns perturbation propagation

Probabilistic heads learn uncertainty

Losses enforce causal and temporal consistency

Outputs (Model Artifacts)
artifacts/models/
├── encoder.pt
├── causal_gnn.pt
├── uncertainty_head.pt
├── decoder.pt
└── metadata.json
Notes
Training is never triggered by the API

Training does not happen at runtime

4. Stage 3 — Validation & Freezing
Purpose
Ensure trained models are reliable, calibrated, and reproducible.

Validation Checks
Predictive performance on held-out data

Uncertainty calibration (ECE, reliability)

Robustness to perturbation noise

Domain shift sanity checks

Relevant Code
evaluation/
experiments/evaluate.py
experiments/ablations.py
Model Freezing
Once validated:

Model weights are frozen

Artifacts are versioned

Metadata records config + dataset hash

Frozen models are now serving-ready.

5. Stage 4 — Model Serving & Simulation (Online)
Purpose
Use trained models to run simulations, not to learn.

How Models Are Loaded
Model paths specified in YAML configs

Mounted into Docker containers

Loaded once at service startup

Example (deployment.yaml)
models:
  encoder: artifacts/models/encoder.pt
  gnn: artifacts/models/causal_gnn.pt
  uncertainty: artifacts/models/uncertainty_head.pt
Runtime Behavior
API receives simulation request

Models perform inference only

No gradients, no weight updates

Deterministic & reproducible behavior

Relevant Code
apps/api/
simulation/
models/
6. Stage 5 — Iteration & Improvement
When Retraining Is Needed
New datasets added

New modalities introduced

Significant performance drift

Research improvements

Iteration Flow
Update data or configs
→ Retrain models offline
→ Revalidate
→ Replace artifacts
→ Redeploy services
Important Rule
Serving systems are never modified to fix model quality.
Model quality is fixed offline.

7. Model Versioning Principles
Models are immutable once deployed

Each version has:

Config hash

Dataset version

Training timestamp

Rollbacks are artifact-based, not code-based

8. What ATLAS Explicitly Avoids
Online training

Silent model updates

Auto-retraining without validation

Black-box model swapping

These practices reduce trust and reproducibility.

9. Summary (One Paragraph)
ATLAS follows a clean, industry-standard ML lifecycle where models are trained offline,
validated rigorously, frozen into immutable artifacts, and served via containerized
services for simulation-only workloads. This separation enables reproducibility,
scalability, and research-grade reliability.


---

# ✅ What *You* Need to Do After All Code Is Written (Step-by-Step)

This is your **exact execution checklist**.

---

## 🔹 Step 0 — Sanity Check
Before anything else:
- Repo structure is complete
- Config files exist
- Training scripts run without syntax errors

---

## 🔹 Step 1 — Prepare Data
```bash
python scripts/preprocess_data.py
Confirm:

data/processed/
graphs/
exist and are populated.

🔹 Step 2 — Train Models (ONCE)
python experiments/train.py --config configs/model.yaml
Confirm:

artifacts/models/
contains .pt files.

🔹 Step 3 — Validate Models
python experiments/evaluate.py
Optional but recommended:

python experiments/ablations.py
🔹 Step 4 — Freeze Models
Do nothing special — freezing means you stop training and treat artifacts as immutable.

(Optional: rename folder)

artifacts/models_v1/
🔹 Step 5 — Configure Deployment
Edit:

configs/deployment.yaml
Point to correct artifact paths.

🔹 Step 6 — Build Containers
docker-compose build
🔹 Step 7 — Run ATLAS
docker-compose up
Open:

UI → http://localhost:3000

API → http://localhost:8000

🔹 Step 8 — Use ATLAS
Run simulations

Compare scenarios

View explanations

Generate recommendations

No retraining happens here.

🔹 Step 9 — Iterate (When Needed)
When improving models:

Stop services

Retrain offline

Replace artifacts

Restart Docker

