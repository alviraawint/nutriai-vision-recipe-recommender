# NutriAI

NutriAI is an AI-powered nutrition assistant that detects likely food ingredients from an uploaded image and recommends relevant recipes from a curated recipe database. The project combines computer vision, TensorFlow model evaluation, SQLite data management, content-based recommendation, and a Streamlit user interface.

## Area

NutriAI was built as an end-to-end AI/Data Science portfolio project. It demonstrates how a trained computer vision model can be connected to a practical recommendation workflow: image upload, ingredient prediction, confidence-based ranking, SQLite recipe retrieval, nutrition-aware scoring, and interactive presentation in Streamlit.

| Area | Evidence in this project |
|---|---|
| AI/ML | EfficientNetB0 transfer learning, image augmentation, class weighting, validation experiments, Top-1/Top-5 evaluation |
| Python | Modular scripts for app runtime, recommendation logic, training, evaluation, and database creation |
| TensorFlow/Keras | Trained image classifier using an ImageNet EfficientNetB0 backbone and custom classification head |
| SQL/SQLite | Local relational recipe database with recipes, normalized ingredients, and recipe-ingredient joins |
| Streamlit | Interactive upload UI, user filters, prediction cards, recommendation cards, and screenshots |
| Recommendation Systems | Content-based recipe ranking using ingredient overlap, model confidence, nutrition tags, and user filters |

## Project Overview

Choosing recipes from available ingredients can be time-consuming, especially when users also care about goals such as low sodium, high protein, lower calories, or lower fat. NutriAI addresses this by using image-based ingredient recognition and interpretable recipe ranking to help users discover recipes from visual input.

## Key Features

- Upload an ingredient image through a Streamlit interface.
- Detect top predicted ingredients with confidence scores.
- Recommend recipes based on detected ingredients and user preferences.
- Filter by health goal, meal type, diet type, and cooking method.
- Store recipe and nutrition data in a local SQLite database.
- Train, fine-tune, and evaluate image classification experiments.
- Report both classification metrics and recommendation-oriented metrics.

## Tech Stack

- **Language:** Python 3.10 or 3.11 recommended
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

The image classifier uses **EfficientNetB0** as the feature extraction backbone with ImageNet weights. A custom classification head is added for the ingredient recognition task using global average pooling, batch normalization, a dense ReLU layer, dropout, and a **softmax multiclass output** trained with categorical cross-entropy.

At inference time, the Streamlit app uses patch-based prediction over the uploaded image and aggregates patch predictions to produce the top likely ingredient labels. These top labels are then used as recommendation signals.

The selected portfolio model is stored at:

```text
models/best_model.keras
```

Class label mappings are stored at:

```text
models/class_indices.npy
```

## Recommendation Approach

NutriAI uses an interpretable content-based ranking approach. The recommender compares detected ingredients against recipe ingredient lists from the SQLite database, applies user-selected filters, and ranks candidate recipes using:

- **Ingredient overlap:** which detected ingredients appear in each recipe.
- **Prediction confidence:** average model confidence for matched ingredients.
- **Ingredient coverage ratio:** how much of a recipe's ingredient list is covered by detections.
- **Nutrition preference score:** whether the recipe aligns with the selected health goal.
- **User filters:** meal type, diet type, and cooking method.

A simplified scoring view is:

```text
final recommendation score =
  weighted ingredient coverage
+ weighted model confidence
+ weighted recipe match ratio
+ weighted nutrition preference
+ small overlap bonus
```

This makes the recommendation logic explainable: each recommendation is tied to matched detected ingredients, nutrition metadata, and selected user preferences.

## Evaluation Metrics

Model and system evaluation results are stored under:

```text
results/training_outputs/
```

Available selected experiment metrics:

| Metric | Score | What it shows |
|---|---:|---|
| Top-1 Accuracy | 0.9749 | The correct ingredient was the highest-ranked prediction. |
| Top-5 Accuracy | 1.0000 | The correct ingredient appeared in the five highest-ranked predictions. |
| HR@5 | 0.6212 | At least one top-5 recommended recipe contained the ground-truth ingredient. |
| NDCG@5 | 0.5982 | Relevant recipes appeared higher in the recommendation ranking. |
| Overall Score | 0.7973 | Weighted end-to-end score combining classification and recommendation metrics. |

The evaluation script combines classifier quality with recommendation quality so the selected model is judged on both prediction accuracy and downstream recipe usefulness.

## Screenshots

### Home Page

Shows the Streamlit landing page, app purpose, and user controls.

![NutriAI home page](screenshots/Home%20page.png)

### Prediction Result

Shows uploaded-image inference with top ingredient predictions and confidence scores.

![NutriAI prediction result](screenshots/Prediction%20result.png)

### Recipe Recommendation Result

Shows nutrition-aware recipe recommendations generated from detected ingredients and user filters.

![NutriAI recipe recommendation result](screenshots/Recipe%20recommendation%20result.png)

## How To Run Locally

1. Clone the repository.

```bash
git clone <repository-url>
cd nutriai-vision-recipe-recommender
```

2. Create and activate a virtual environment. Python 3.10 or 3.11 is recommended for TensorFlow compatibility.

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
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

To rerun training experiments:

```bash
python train_model.py
```

To rerun model and recommendation evaluation:

```bash
python evaluate.py
```

## Folder Structure

```text
nutriai-vision-recipe-recommender/
  app.py                       # Streamlit application and image inference flow
  recommender.py               # Content-based recipe recommendation logic
  train_model.py               # TensorFlow/EfficientNetB0 training pipeline
  evaluate.py                  # Classification + recommendation evaluation script
  requirements.txt             # Pinned Python dependencies
  README.md                    # Main portfolio documentation
  database/
    create_database.py         # Builds SQLite database from recipes.csv
    recipes.csv                # Curated recipe and nutrition data
    nutriai.db                 # Local SQLite runtime database
  models/
    best_model.keras           # Selected trained Keras model
    class_indices.npy          # Ingredient class label mapping
  results/
    training_outputs/          # Metrics, plots, manifests, and experiment logs
  screenshots/                 # README/demo screenshots
  sample_images/               # Optional sample test images
  docs/                        # Additional project and folder documentation
```

For a more detailed explanation of each folder, see [`docs/folder_structure.md`](docs/folder_structure.md).

## Limitations

- Recommendations depend on the quality and coverage of the curated recipe database.
- Ingredient detection performance may decrease for unclear, cluttered, or multi-object images.
- The current classifier is trained as a multiclass ingredient recognizer; complex multi-ingredient scenes are approximated with patch-based inference and top-k ranking.
- Nutrition values are sourced from the project dataset and are not medical advice.
- The current app runs locally and is not optimized for large-scale deployment.
- The dataset is kept local/archived to keep the public portfolio lightweight.

## Future Improvements

- Add more diverse ingredient images and recipe categories.
- Improve explicit multi-ingredient detection for complex food scenes.
- Add explainability visualizations for model predictions.
- Deploy the app using Streamlit Community Cloud or another hosting platform.
- Add automated tests for recommendation scoring and database integrity.
- Expand nutrition personalization with user profiles and dietary constraints.
