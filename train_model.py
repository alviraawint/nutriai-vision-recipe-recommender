import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
if not DATASET_DIR.exists():
    DATASET_DIR = BASE_DIR / "archive_unused" / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
OUTPUT_DIR = BASE_DIR / "results" / "training_outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

IMG_SIZE = 224
INPUT_SIZE = (IMG_SIZE, IMG_SIZE)
EPOCHS = 30
MONITOR = "val_accuracy"
RESUME_COMPLETED = True
TRAINING_VERSION = "speed_accuracy_v2"
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3

BASELINE_LR = 3e-4
BASELINE_BATCH_SIZE = 32
BASELINE_DROPOUT = 0.3

LEARNING_RATES = [1e-3, 3e-4, 1e-4]
BATCH_SIZES = [16, 32, 64]
DROPOUT_RATES = [0.2, 0.3, 0.5]


# =============================================================================
# SMALL HELPERS
# =============================================================================
def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def lr_tag(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def dropout_tag(value):
    return str(value).replace(".", "p")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def best_epoch_value(history, metric):
    values = history.get(metric, [])
    return float(max(values)) if values else 0.0


def same_config(saved_result, cfg):
    return (
        saved_result.get("training_version") == TRAINING_VERSION
        and saved_result.get("learning_rate") == cfg["learning_rate"]
        and saved_result.get("batch_size") == cfg["batch_size"]
        and saved_result.get("dropout_rate") == cfg["dropout_rate"]
    )


def completed_result_for(cfg):
    """Return saved result if this experiment is complete and matches cfg."""
    tag = cfg["tag"]
    result_path = OUTPUT_DIR / f"result_{tag}.json"
    best_model_path = OUTPUT_DIR / f"best_{tag}.keras"
    final_model_path = OUTPUT_DIR / f"final_{tag}.keras"
    history_path = OUTPUT_DIR / f"history_{tag}.json"

    if not RESUME_COMPLETED:
        return None

    if not (
        result_path.exists()
        and best_model_path.exists()
        and final_model_path.exists()
        and history_path.exists()
    ):
        return None

    try:
        result = load_json(result_path)
    except (json.JSONDecodeError, OSError):
        return None

    if not same_config(result, cfg):
        print(f"[INFO] Existing result for {tag} uses different settings. Retraining.")
        return None

    result["resumed_from_disk"] = True
    return result


def save_manifest(manifest, final=False):
    filename = "manifest.json" if final else "manifest_partial.json"
    save_json(OUTPUT_DIR / filename, manifest)


# =============================================================================
# GENERATOR BUILDER
# =============================================================================
def build_generators(batch_size):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.08,
        height_shift_range=0.08,
        brightness_range=[0.85, 1.15],
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=INPUT_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


def compute_class_weights(train_gen):
    class_counts = np.bincount(train_gen.classes)
    total_samples = len(train_gen.classes)
    num_classes = len(class_counts)

    class_weights = {}
    for class_idx, count in enumerate(class_counts):
        if count == 0:
            class_weights[class_idx] = 1.0
        else:
            class_weights[class_idx] = float(total_samples / (num_classes * count))
    return class_weights


# =============================================================================
# MODEL BUILDER
# =============================================================================
def build_model(num_classes, learning_rate, dropout_rate):
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model


# =============================================================================
# PLOTS
# =============================================================================
def save_plots(history, tag):
    epochs_range = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs_range, history.get("val_accuracy", []), label="Val Accuracy")
    if "top5_accuracy" in history:
        plt.plot(epochs_range, history.get("top5_accuracy", []), label="Train Top-5")
    if "val_top5_accuracy" in history:
        plt.plot(epochs_range, history.get("val_top5_accuracy", []), label="Val Top-5")
    plt.title(f"Accuracy - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"accuracy_{tag}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("loss", []), label="Train Loss")
    plt.plot(epochs_range, history.get("val_loss", []), label="Val Loss")
    plt.title(f"Loss - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"loss_{tag}.png", dpi=150)
    plt.close()


# =============================================================================
# TRAIN ONE EXPERIMENT
# =============================================================================
def train_one_experiment(phase_name, cfg):
    tag = cfg["tag"]
    learning_rate = cfg["learning_rate"]
    batch_size = cfg["batch_size"]
    dropout_rate = cfg["dropout_rate"]

    completed_result = completed_result_for(cfg)
    if completed_result is not None:
        print("\n" + "=" * 72)
        print(f"PHASE: {phase_name}")
        print(f"EXPERIMENT: {tag}")
        print(f"LR={learning_rate} | BS={batch_size} | Dropout={dropout_rate}")
        print("[RESUME] Completed experiment found. Skipping training.")
        print("=" * 72)
        return completed_result

    print("\n" + "=" * 72)
    print(f"PHASE: {phase_name}")
    print(f"EXPERIMENT: {tag}")
    print(f"LR={learning_rate} | BS={batch_size} | Dropout={dropout_rate}")
    print("=" * 72)

    train_gen, val_gen = build_generators(batch_size)
    num_classes = len(train_gen.class_indices)
    class_weights = compute_class_weights(train_gen)

    class_indices_path = OUTPUT_DIR / "class_indices.npy"
    if not class_indices_path.exists():
        np.save(class_indices_path, train_gen.class_indices)
        print(f"[INFO] Saved class indices: {class_indices_path}")
    print("[INFO] Using class weights to reduce class imbalance impact.")

    model = build_model(
        num_classes=num_classes,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    summary_path = OUTPUT_DIR / f"summary_{tag}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    best_model_path = OUTPUT_DIR / f"best_{tag}.keras"
    final_model_path = OUTPUT_DIR / f"final_{tag}.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=MONITOR,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history_obj = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    model.save(final_model_path)

    history = {
        key: [float(v) for v in values]
        for key, values in history_obj.history.items()
    }
    save_json(OUTPUT_DIR / f"history_{tag}.json", history)
    save_plots(history, tag)

    result = {
        "phase": phase_name,
        "tag": tag,
        "training_version": TRAINING_VERSION,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "best_val_accuracy": best_epoch_value(history, "val_accuracy"),
        "best_val_top5_accuracy": best_epoch_value(history, "val_top5_accuracy"),
        "best_val_loss": float(min(history.get("val_loss", [999.0]))),
        "epochs_trained": len(history.get("loss", [])),
    }

    save_json(OUTPUT_DIR / f"result_{tag}.json", result)
    print(f"[INFO] Saved final model: {final_model_path}")
    print(f"[INFO] Best validation accuracy: {result['best_val_accuracy']:.4f}")
    print(f"[INFO] Best validation Top-5: {result['best_val_top5_accuracy']:.4f}")
    return result


def choose_best_by_validation(results):
    return max(
        results,
        key=lambda item: (
            item["best_val_accuracy"],
            item.get("best_val_top5_accuracy", 0.0),
            -item["best_val_loss"],
        ),
    )


# =============================================================================
# PHASE-BY-PHASE TRAINING
# =============================================================================
def main():
    ensure_dirs()

    all_results = []
    manifest = {
        "seed": SEED,
        "resume_completed": RESUME_COMPLETED,
        "training_version": TRAINING_VERSION,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lr_reduce_patience": LR_REDUCE_PATIENCE,
        "augmentation_note": (
            "Moderate augmentation is used to improve robustness without "
            "over-distorting ingredient images."
        ),
        "class_weighting": True,
        "metrics": ["accuracy", "top5_accuracy"],
        "selection_note": (
            "Training phases use validation accuracy and validation loss to "
            "decide the next training setting. Final model selection should "
            "be done with evaluate.py using detection and recommendation metrics."
        ),
        "baseline": {
            "learning_rate": BASELINE_LR,
            "batch_size": BASELINE_BATCH_SIZE,
            "dropout_rate": BASELINE_DROPOUT,
        },
        "phases": {},
        "experiments": [],
    }

    phase1_configs = [
        {
            "tag": f"p1_lr_{lr_tag(lr)}",
            "learning_rate": lr,
            "batch_size": BASELINE_BATCH_SIZE,
            "dropout_rate": BASELINE_DROPOUT,
        }
        for lr in LEARNING_RATES
    ]
    phase1_results = [train_one_experiment("learning_rate", cfg) for cfg in phase1_configs]
    best_lr_result = choose_best_by_validation(phase1_results)
    best_lr = best_lr_result["learning_rate"]
    all_results.extend(phase1_results)
    manifest["phases"]["learning_rate"] = {
        "tested": phase1_results,
        "chosen": best_lr_result,
    }
    manifest["experiments"] = all_results
    save_manifest(manifest)

    phase2_configs = [
        {
            "tag": f"p2_bs_{bs}",
            "learning_rate": best_lr,
            "batch_size": bs,
            "dropout_rate": BASELINE_DROPOUT,
        }
        for bs in BATCH_SIZES
    ]
    phase2_results = [train_one_experiment("batch_size", cfg) for cfg in phase2_configs]
    best_bs_result = choose_best_by_validation(phase2_results)
    best_bs = best_bs_result["batch_size"]
    all_results.extend(phase2_results)
    manifest["phases"]["batch_size"] = {
        "tested": phase2_results,
        "chosen": best_bs_result,
    }
    manifest["experiments"] = all_results
    save_manifest(manifest)

    phase3_configs = [
        {
            "tag": f"p3_do_{dropout_tag(dropout)}",
            "learning_rate": best_lr,
            "batch_size": best_bs,
            "dropout_rate": dropout,
        }
        for dropout in DROPOUT_RATES
    ]
    phase3_results = [train_one_experiment("dropout_rate", cfg) for cfg in phase3_configs]
    best_dropout_result = choose_best_by_validation(phase3_results)
    all_results.extend(phase3_results)
    manifest["phases"]["dropout_rate"] = {
        "tested": phase3_results,
        "chosen": best_dropout_result,
    }

    manifest["experiments"] = all_results
    manifest["validation_selected_config"] = {
        "learning_rate": best_lr,
        "batch_size": best_bs,
        "dropout_rate": best_dropout_result["dropout_rate"],
        "best_model_path": best_dropout_result["best_model_path"],
        "final_model_path": best_dropout_result["final_model_path"],
    }

    save_manifest(manifest, final=True)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Best LR by validation       : {best_lr}")
    print(f"Best batch size by validation: {best_bs}")
    print(f"Best dropout by validation  : {best_dropout_result['dropout_rate']}")
    print(f"Outputs saved in            : {OUTPUT_DIR}")
    print("Run evaluate.py for final end-to-end model selection.")


if __name__ == "__main__":
    main()
