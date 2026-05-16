import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from recommender import load_recipes, normalise_set, recommend_recipes


# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR / "archive_unused" / "dataset"
TEST_DIR = DATASET_DIR / "test"
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "training_outputs"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "results.json"
DB_PATH = BASE_DIR / "database" / "nutriai.db"

IMG_SIZE = 224
INPUT_SIZE = (IMG_SIZE, IMG_SIZE)
TEST_BATCH_SIZE = 32
K = 5

# Balanced, simple end-to-end score.
METRIC_WEIGHTS = {
    "top1": 0.30,
    "top5": 0.20,
    "hr5": 0.25,
    "ndcg5": 0.25,
}


# =============================================================================
# JSON HELPER
# =============================================================================
def save_json(path, data):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_path(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else BASE_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NutriAI models without overwriting results unless requested."
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder containing models to evaluate when no manifest is used.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest JSON to load experiments from. Use --no-manifest to ignore it.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Ignore manifest and scan --models-dir for models.",
    )
    parser.add_argument(
        "--pattern",
        default="best_*.keras",
        help="Model filename pattern used with --no-manifest, e.g. final_bs_32.keras.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_RESULTS_PATH),
        help="Where to save evaluation results.",
    )
    return parser.parse_args()


# =============================================================================
# LOAD EXPERIMENTS
# =============================================================================
def load_manifest(models_dir, manifest_path, use_manifest=True, pattern="best_*.keras"):
    if use_manifest and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        experiments = manifest.get("experiments", [])
        known_paths = {
            str(Path(exp.get("best_model_path") or exp.get("final_model_path", "")))
            for exp in experiments
        }
        for result_path in sorted(models_dir.glob("fine_tune_result_*.json")):
            with open(result_path, "r", encoding="utf-8-sig") as f:
                fine_tune_result = json.load(f)
            fine_tune_path = str(Path(fine_tune_result["best_model_path"]))
            if fine_tune_path not in known_paths:
                experiments.append(fine_tune_result)
                known_paths.add(fine_tune_path)
        manifest["experiments"] = experiments
        return manifest

    experiments = []
    for model_path in sorted(models_dir.glob(pattern)):
        tag = model_path.stem
        if tag.startswith("best_"):
            tag = tag.replace("best_", "", 1)
        elif tag.startswith("final_"):
            tag = tag.replace("final_", "", 1)
        experiments.append(
            {
                "phase": "unknown",
                "tag": tag,
                "learning_rate": None,
                "batch_size": None,
                "dropout_rate": None,
                "best_model_path": str(model_path),
                "final_model_path": str(model_path),
            }
        )
    return {"experiments": experiments, "phases": {}}


# =============================================================================
# TEST GENERATOR
# =============================================================================
def build_test_generator():
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=INPUT_SIZE,
        batch_size=TEST_BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )


# =============================================================================
# DETECTION METRICS
# =============================================================================
def compute_topk_accuracy(y_true, y_probs, k=5):
    top1_correct = 0
    topk_correct = 0

    for true_idx, probs in zip(y_true, y_probs):
        top_k = np.argsort(probs)[-k:][::-1]
        if top_k[0] == true_idx:
            top1_correct += 1
        if true_idx in top_k:
            topk_correct += 1

    total = len(y_true)
    return top1_correct / total, topk_correct / total


# =============================================================================
# RECOMMENDATION METRICS
# =============================================================================
def recipe_contains_ingredient(recipe, ingredient):
    recipe_ingredients = normalise_set(recipe.get("ingredient_list", []))
    return ingredient in recipe_ingredients


def build_recipe_lookup(db_path):
    recipes = load_recipes(str(db_path))
    return {recipe["recipe_name"]: recipe for recipe in recipes}


def compute_recommendation_metrics(
    y_true,
    y_probs,
    idx_to_class,
    recipe_lookup,
    k=5,
):
    hr_scores = []
    ndcg_scores = []

    for true_idx, probs in zip(y_true, y_probs):
        top_indices = np.argsort(probs)[-k:][::-1]
        detected_ingredients = [idx_to_class[int(i)] for i in top_indices]
        detected_confidences = {
            idx_to_class[int(i)]: float(probs[int(i)])
            for i in top_indices
        }
        true_ingredient = normalise_set([idx_to_class[int(true_idx)]]).pop()

        recommendations = recommend_recipes(
            detected_ingredients=detected_ingredients,
            detected_confidences=detected_confidences,
            health_goal="Balanced",
            meal_filter="Any",
            diet_filter="Any",
            cooking_method_filter="Any",
            top_n=k,
            min_overlap=1,
            db_path=str(DB_PATH),
        )

        hit_rank = None
        for rank, rec in enumerate(recommendations, start=1):
            recipe = recipe_lookup.get(rec["recipe_name"])
            if recipe and recipe_contains_ingredient(recipe, true_ingredient):
                hit_rank = rank
                break

        if hit_rank is None:
            hr_scores.append(0.0)
            ndcg_scores.append(0.0)
        else:
            hr_scores.append(1.0)
            ndcg_scores.append(1.0 / math.log2(hit_rank + 1))

    return float(np.mean(hr_scores)), float(np.mean(ndcg_scores))


def compute_overall_score(top1, top5, hr5, ndcg5):
    return (
        METRIC_WEIGHTS["top1"] * top1
        + METRIC_WEIGHTS["top5"] * top5
        + METRIC_WEIGHTS["hr5"] * hr5
        + METRIC_WEIGHTS["ndcg5"] * ndcg5
    )


# =============================================================================
# EVALUATE ONE MODEL
# =============================================================================
def evaluate_model(experiment, test_gen, idx_to_class, recipe_lookup):
    selected_path = experiment.get("best_model_path") or experiment.get("final_model_path")
    model_source = "best_model_path" if experiment.get("best_model_path") else "final_model_path"
    model_path = Path(selected_path)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    tag = experiment["tag"]
    print("\n" + "-" * 72)
    print(f"Evaluating {tag}")
    print(f"Model: {model_path} ({model_source})")

    if not model_path.exists():
        print(f"[SKIP] Missing model: {model_path}")
        return None

    model = tf.keras.models.load_model(model_path)
    test_gen.reset()

    steps = math.ceil(test_gen.samples / test_gen.batch_size)
    all_probs = []
    all_labels = []

    for _ in range(steps):
        batch_x, batch_y = next(test_gen)
        probs = model.predict(batch_x, verbose=0)
        all_probs.append(probs)
        all_labels.append(np.argmax(batch_y, axis=1))

    y_probs = np.vstack(all_probs)[: test_gen.samples]
    y_true = np.concatenate(all_labels)[: test_gen.samples]

    top1, top5 = compute_topk_accuracy(y_true, y_probs, k=K)
    hr5, ndcg5 = compute_recommendation_metrics(
        y_true=y_true,
        y_probs=y_probs,
        idx_to_class=idx_to_class,
        recipe_lookup=recipe_lookup,
        k=K,
    )
    overall_score = compute_overall_score(top1, top5, hr5, ndcg5)

    result = {
        "phase": experiment.get("phase"),
        "tag": tag,
        "learning_rate": experiment.get("learning_rate"),
        "batch_size": experiment.get("batch_size"),
        "dropout_rate": experiment.get("dropout_rate"),
        "model_source": model_source,
        "model_path": str(model_path),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "hr_at_5": round(hr5, 4),
        "ndcg_at_5": round(ndcg5, 4),
        "overall_score": round(overall_score, 4),
    }

    print(f"Top-1 Accuracy : {top1:.4f}")
    print(f"Top-5 Accuracy : {top5:.4f}")
    print(f"HR@5           : {hr5:.4f}")
    print(f"NDCG@5         : {ndcg5:.4f}")
    print(f"Overall Score  : {overall_score:.4f}")
    return result


def choose_best(results):
    return max(
        results,
        key=lambda item: (
            item["overall_score"],
            item["ndcg_at_5"],
            item["hr_at_5"],
            item["top1_accuracy"],
        ),
    )


def best_by_phase(results, phase_name):
    phase_results = [item for item in results if item.get("phase") == phase_name]
    return choose_best(phase_results) if phase_results else None


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    models_dir = resolve_path(args.models_dir)
    manifest_path = resolve_path(args.manifest)
    results_path = resolve_path(args.output_json)
    models_dir.mkdir(exist_ok=True)

    manifest = load_manifest(
        models_dir=models_dir,
        manifest_path=manifest_path,
        use_manifest=not args.no_manifest,
        pattern=args.pattern,
    )
    experiments = manifest.get("experiments", [])
    if not experiments:
        raise RuntimeError("No experiment models found to evaluate.")

    test_gen = build_test_generator()
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    recipe_lookup = build_recipe_lookup(DB_PATH)

    all_results = []
    for experiment in experiments:
        result = evaluate_model(experiment, test_gen, idx_to_class, recipe_lookup)
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No models were evaluated successfully.")

    best_lr = best_by_phase(all_results, "learning_rate")
    best_bs = best_by_phase(all_results, "batch_size")
    best_dropout = best_by_phase(all_results, "dropout_rate")
    final_best = choose_best(all_results)

    output = {
        "metric_weights": METRIC_WEIGHTS,
        "selection_rule": (
            "Highest weighted end-to-end score, with NDCG@5, HR@5, and Top-1 "
            "Accuracy used as tie breakers."
        ),
        "all_results": all_results,
        "best_learning_rate_experiment": best_lr,
        "best_batch_size_experiment": best_bs,
        "best_dropout_experiment": best_dropout,
        "final_best_model": final_best,
    }

    save_json(results_path, output)

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    if best_lr:
        print(f"Best LR       : {best_lr['learning_rate']} ({best_lr['tag']})")
    if best_bs:
        print(f"Best BS       : {best_bs['batch_size']} ({best_bs['tag']})")
    if best_dropout:
        print(f"Best Dropout  : {best_dropout['dropout_rate']} ({best_dropout['tag']})")
    print(f"Final Best    : {final_best['tag']}")
    print(f"Final Model   : {final_best['model_path']}")
    print(f"Overall Score : {final_best['overall_score']:.4f}")
    print(f"Results saved : {results_path}")


if __name__ == "__main__":
    main()
