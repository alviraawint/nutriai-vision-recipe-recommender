# NutriAI

NutriAI is an AI-powered nutrition assistant that detects food ingredients from an uploaded image and recommends relevant recipes from a curated recipe database. The project combines computer vision, model evaluation, SQLite data management, and content-based recommendation in a Streamlit application.

## Project Overview

This project was built as an end-to-end AI/Data Science portfolio project. It demonstrates how an image classification model can be connected to a practical recommendation workflow: predict ingredients, map predictions to recipe ingredients, rank candidate recipes, and present results through an interactive web interface.

## Problem Statement

Choosing recipes from available ingredients can be time-consuming, especially when users also care about health goals such as low sodium, high protein, or lower calories. NutriAI addresses this by using ingredient image detection and recipe ranking to help users discover suitable recipes from visual input.

## Key Features

- Upload an ingredient image through a Streamlit interface
- Detect top predicted ingredients with confidence scores
- Recommend recipes based on detected ingredients and user preferences
- Filter by health goal, meal type, diet type, and cooking method
- Store recipe and nutrition data in a local SQLite database
- Train, fine-tune, and evaluate image classification experiments
- Report both classification and recommendation-oriented metrics

## Tech Stack

- **Language:** Python
- **App Framework:** Streamlit
- **Deep Learning:** TensorFlow, Keras, EfficientNetB0
- **Data Processing:** NumPy, Pandas
- **Database:** SQLite
- **Visualization:** Matplotlib
- **Image Handling:** Pillow

## System Workflow

```text
User uploads image
        |
        v
Image preprocessing and patch-based prediction
        |
        v
EfficientNetB0 ingredient classifier
        |
        v
Top ingredient labels + confidence scores
        |
        v
Content-based recipe ranking
        |
        v
Filtered recipe recommendations in Streamlit
```

## Model Architecture Summary

The image classifier uses **EfficientNetB0** as the feature extraction backbone. A custom classification head was added for the ingredient recognition task, using global pooling, normalization/dropout layers, and a modified sigmoid-style multi-label prediction head suitable for ranking likely ingredient classes.

The selected portfolio model is stored at:

```text
models/best_model.keras
```

Class label mappings are stored at:

```text
models/class_indices.npy
```

## Recommendation Approach

NutriAI uses a content-based ranking approach. The recommender compares detected ingredients against recipe ingredient lists from the SQLite database, then ranks recipes using:

- ingredient overlap
- prediction confidence
- ingredient coverage ratio
- nutrition preference scoring
- user-selected filters

This makes the recommendation logic interpretable: each recommendation is tied to detected ingredients and matching recipe metadata.

## Evaluation Metrics

Model and system evaluation results are stored under:

```text
results/training_outputs/
```

Available selected experiment metrics:

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 0.9749 |
| Top-5 Accuracy | 1.0000 |
| HR@5 | 0.6212 |
| NDCG@5 | 0.5982 |
| Overall Score | 0.7973 |

## Screenshots

### Home Page

![NutriAI home page](screenshots/Home%20page.png)

### Prediction Result

![NutriAI prediction result](screenshots/Prediction%20result.png)

### Recipe Recommendation Result

![NutriAI recipe recommendation result](screenshots/Recipe%20recommendation%20result.png)

## How To Run Locally

1. Clone the repository.

```bash
git clone <repository-url>
cd nutriai
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app.

```bash
streamlit run app.py
```

Required runtime files:

- `models/best_model.keras`
- `models/class_indices.npy`
- `database/nutriai.db`

To rebuild the database:

```bash
python database/create_database.py
```

## Folder Structure

```text
nutriai/
  app.py                       # Streamlit application
  recommender.py               # Content-based recipe recommendation logic
  train_model.py               # Model training pipeline
  evaluate.py                  # Evaluation script
  requirements.txt             # Python dependencies
  README.md                    # Project documentation
  database/
    create_database.py         # Builds SQLite database from CSV
    recipes.csv                # Curated recipe data
    nutriai.db                 # SQLite database
  models/
    best_model.keras           # Selected trained model
    class_indices.npy          # Class label mapping
  results/
    training_outputs/          # Metrics, plots, and experiment logs
  screenshots/                 # README/demo screenshots
  sample_images/               # Sample test images
  docs/                        # Additional project notes
  archive_unused/              # Archived local/prototype files
```

## Limitations

- Recommendations depend on the quality and coverage of the curated recipe database.
- Ingredient detection performance may decrease for unclear, cluttered, or multi-object images.
- Nutrition values are sourced from the project dataset and are not medical advice.
- The current app runs locally and is not optimized for large-scale deployment.
- The dataset is kept local/archived to keep the public portfolio lightweight.

## Future Improvements

- Add more diverse ingredient images and recipe categories
- Improve multi-ingredient detection for complex food scenes
- Add explainability visualizations for model predictions
- Deploy the app using Streamlit Community Cloud or another hosting platform
- Add automated tests for recommendation scoring and database integrity
- Expand nutrition personalization with user profiles and dietary constraints
