# Project Overview

NutriAI combines image classification, SQLite-backed recipe data, and content-based recommendation in a Streamlit application.

The app loads a trained Keras model, predicts likely ingredients from an uploaded image, then passes the detected ingredients into `recommender.py`. The recommender loads recipes from SQLite and ranks matches using ingredient coverage, model confidence, nutrition goal scoring, and user-selected filters.

## Main Files

- `app.py`: Streamlit user interface, image upload flow, model loading, patch-based prediction, and recommendation display.
- `recommender.py`: recipe loading, ingredient normalization, preference filtering, scoring, and ranking.
- `train_model.py`: EfficientNetB0 transfer-learning experiments with augmentation, class weighting, and validation-based model selection.
- `evaluate.py`: classification metrics plus recommendation metrics such as HR@5 and NDCG@5.
- `database/create_database.py`: rebuilds `database/nutriai.db` from `database/recipes.csv`.

## Documentation Map

- `README.md`: recruiter-friendly project summary, setup instructions, metrics, screenshots, and folder overview.
- `docs/folder_structure.md`: detailed explanation of repository folders and key artifacts.
- `docs/cleanup_notes.md`: notes about how the public portfolio repository was organized.
