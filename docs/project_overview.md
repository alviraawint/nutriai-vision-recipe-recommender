# Project Overview

NutriAI combines image classification and rule-based recommendation.

The Streamlit app loads a trained Keras model, predicts likely ingredients from an uploaded image, then passes the detected ingredients into `recommender.py`. The recommender loads recipes from SQLite and ranks matches using ingredient coverage, model confidence, nutrition goal scoring, and user-selected filters.

## Main Files

- `app.py`: user interface and image prediction flow
- `recommender.py`: recipe loading, normalization, scoring, and ranking
- `train_model.py`: EfficientNetB0 training experiments
- `evaluate.py`: classification and recommendation evaluation
- `database/create_database.py`: rebuilds `database/nutriai.db` from `database/recipes.csv`
