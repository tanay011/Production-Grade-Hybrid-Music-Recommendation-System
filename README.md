# 🎵 Production-Grade Hybrid Music Recommendation System

A production-grade hybrid music recommendation system that combines **content-based filtering** and **collaborative filtering** to deliver personalized song recommendations — with a full MLOps pipeline for reproducible training and zero-downtime deployment.

> **EGN 6217 · Spring 2026 · Kunal Jagtap**

---

## 📌 Project Overview

Music streaming platforms serve hundreds of millions of users, yet personalized recommendation remains a persistent challenge. This project addresses it by:

- **Fusing two complementary ML signals** — audio feature similarity (content-based) and crowd listening behavior (collaborative filtering) — into a single hybrid score
- **Exposing user control** via a diversity slider that blends content and collaborative signals in real time
- **Handling cold-start gracefully** — tracks with no listening history fall back to content-only mode, ensuring 100% catalog coverage
- **Operationalizing with full MLOps** — DVC, Docker, GitHub Actions, Amazon ECR, AWS CodeDeploy, EC2 Auto Scaling Group + ALB

The end result is a **Streamlit web app** where users search any song, pick the artist, tune a diversity slider, and instantly receive ranked recommendations — each with an embedded Spotify 30-second audio preview.

---

## 🗂️ Repository Structure

```
spotify-hybrid-recommender/
│
├── data/                        # Raw and processed data (DVC-tracked, not in Git)
│   ├── Music_Info.csv
│   └── User_Listening_History.csv
│
├── src/                         # Core pipeline scripts
│   ├── data_cleaning.py         # Stage 1: dedup, drop cols, lowercase
│   ├── content_based_filtering.py  # Stage 2: CountEnc, OHE, TF-IDF, Scaler
│   ├── collaborative_filtering.py  # Stage 3: Dask aggregation, CSR matrix
│   └── transform_filtered_data.py  # Stage 4: apply transformer to collab subset
│
├── notebooks/
│   └── setup.ipynb              # Environment verification & data exploration
│
├── ui/
│   └── app.py                   # Stage 5: Streamlit app (inference + UI)
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [DVC](https://dvc.org/) with S3 remote support
- AWS credentials configured (for DVC data pull)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/spotify-hybrid-recommender.git
cd spotify-hybrid-recommender
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Pull Data from DVC Remote (S3)

```bash
dvc pull
```

> Requires AWS credentials with access to the private S3 bucket. For CI/CD, credentials are stored as GitHub Secrets.

### 4. Run the Full Pipeline

```bash
dvc repro
```

This executes all five stages in order:
1. `data_cleaning.py` → `cleaned_data.csv`
2. `content_based_filtering.py` → `transformed_data.npz` + `transformer.joblib`
3. `collaborative_filtering.py` → `interaction_matrix.npz` + `track_ids.npy` + `collab_filtered_data.csv`
4. `transform_filtered_data.py` → `transformed_hybrid_data.npz`
5. `app.py` (inference at query time)

---

## 📊 Dataset

| Dataset | Source | Size | Key Columns |
|---|---|---|---|
| `Music_Info.csv` | Spotify API export | ~170,000 tracks | `track_id`, `name`, `artist`, `year`, `danceability`, `energy`, `tempo`, `loudness`, `key`, `time_signature`, `tags`, `spotify_preview_url` |
| `User_Listening_History.csv` | Last.fm-style export | Millions of rows | `user_id`, `track_id`, `playcount` |

> ⚠️ **Data is not committed to Git.** It is DVC-tracked and stored in a private S3 bucket. Run `dvc pull` to download.

---

## 🏗️ Architecture

### Hybrid Scoring Formula

```
hybrid_score = w · normalize(content_cosine) + (1 − w) · normalize(collab_cosine)
```

Where `w = 1 − (slider_value / 10)` — controlled by the user's diversity slider at inference time.

### ML Pipeline (DVC Stages)

```
Raw Data (S3/DVC)
      ↓
Stage 1 — data_cleaning.py         → cleaned_data.csv
      ↓
Stage 2 — content_based_filtering  → transformed_data.npz + transformer.joblib
      ↓
Stage 3 — collaborative_filtering  → interaction_matrix.npz + collab_filtered_data.csv
      ↓
Stage 4 — transform_filtered_data  → transformed_hybrid_data.npz
      ↓
Stage 5 — Inference (app.py)       → weighted hybrid score → top-k results
      ↓
Streamlit Interface
```

---

## 🔁 MLOps Stack

| Tool | Purpose |
|---|---|
| DVC | Data versioning + reproducible pipeline |
| Docker | Containerized, environment-consistent deployment |
| GitHub Actions | CI/CD — build, test, push to ECR on every commit |
| Amazon ECR | Container registry |
| AWS CodeDeploy | Blue/green zero-downtime deployment |
| EC2 ASG + ALB | Auto-scaling compute + load balancing |

---

## 👤 Author

**Kunal Jagtap**
EGN 6217 — Machine Learning Systems · Spring 2026
University of Florida
