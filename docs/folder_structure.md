# Folder Structure Guide

This document explains how the NutriAI repository is organized and which files are important for recruiters, reviewers, and future maintainers.

## Root Files

| Path | Purpose |
|---|---|
| `README.md` | Main portfolio-facing documentation with project summary, setup instructions, metrics, screenshots, and limitations. |
| `app.py` | Streamlit application. Handles UI rendering, model loading, image preprocessing, patch-based prediction, and recommendation display. |
| `recommender.py` | Content-based recommendation module. Loads recipes from SQLite, normalizes ingredients, applies filters, scores recipes, and returns ranked recommendations. |
| `train_model.py` | TensorFlow/EfficientNetB0 training pipeline. Runs augmentation, class weighting, hyperparameter experiments, checkpointing, and plot generation. |
| `evaluate.py` | Evaluation script that reports both classifier metrics and recommendation metrics. |
| `requirements.txt` | Pinned runtime dependencies for local reproduction. |

## `database/`

| Path | Purpose |
|---|---|
| `database/create_database.py` | Rebuilds the SQLite database from the curated CSV file. |
| `database/recipes.csv` | Curated recipe, ingredient, nutrition, meal tag, diet type, and cooking method data. |
| `database/nutriai.db` | Local SQLite database used by the Streamlit app and recommender. |

The database is organized around recipe records and normalized ingredient relationships so the recommender can compare detected ingredients against recipe ingredient lists.

## `models/`

| Path | Purpose |
|---|---|
| `models/best_model.keras` | Selected trained Keras model used by the Streamlit app. |
| `models/class_indices.npy` | Class-index mapping used to convert model output indices back into ingredient labels. |

These files are required to run inference locally.

## `results/`

| Path | Purpose |
|---|---|
| `results/training_outputs/` | Training histories, experiment result JSON files, selected-model metadata, and generated plots. |
| `results/training_outputs/plots/` | Accuracy and loss visualizations from training experiments. |

This folder provides evidence of model experimentation and evaluation beyond the deployed app.

## `screenshots/`

Contains screenshots used by the README to show the Streamlit home page, prediction output, and recipe recommendation output.

## `sample_images/`

Reserved for lightweight sample inputs that can be used to manually test the app. Keeping sample images here makes it easier for reviewers to reproduce the demo workflow.

## `docs/`

| Path | Purpose |
|---|---|
| `docs/project_overview.md` | Concise technical overview and documentation map. |
| `docs/folder_structure.md` | Detailed folder and artifact guide. |
| `docs/cleanup_notes.md` | Notes about repository cleanup for public portfolio use. |

## Suggested Review Path

For a quick recruiter or technical review, read files in this order:

1. `README.md`
2. `screenshots/`
3. `recommender.py`
4. `train_model.py`
5. `evaluate.py`
6. `database/create_database.py`
7. `results/training_outputs/results.json`
