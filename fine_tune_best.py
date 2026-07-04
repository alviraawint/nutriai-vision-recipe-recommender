import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import BatchNormalization
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
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_PATH = OUTPUT_DIR / "results.json"
PLOTS_DIR = OUTPUT_DIR / "plots"

IMG_SIZE = 224
INPUT_SIZE = (IMG_SIZE, IMG_SIZE)
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5
FINE_TUNE_LAST_LAYERS = 30
DEFAULT_BATCH_SIZE = 32


# =============================================================================
# HELPERS
# =============================================================================
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_best_experiment():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "results.json not found. Run evaluate.py before fine-tuning."
        )

    with open(RESULTS_PATH, "r", encoding="utf-8-sig") as f:
        results = json.load(f)

    best = results.get("final_best_model")
    if not best:
        raise RuntimeError("final_best_model was not found in results.json.")
    return best


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


def unfreeze_last_layers(model, last_layers):
    for layer in model.layers:
        layer.trainable = False

    trainable_count = 0
    for layer in reversed(model.layers):
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
            continue
        layer.trainable = True
        trainable_count += 1
        if trainable_count >= last_layers:
            break

    return trainable_count


def save_plots(history, tag):
    PLOTS_DIR.mkdir(exist_ok=True)
    epochs_range = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("accuracy", []), label="Train Accuracy")
    plt.plot(epochs_range, history.get("val_accuracy", []), label="Val Accuracy")
    if "top5_accuracy" in history:
        plt.plot(epochs_range, history.get("top5_accuracy", []), label="Train Top-5")
    if "val_top5_accuracy" in history:
        plt.plot(epochs_range, history.get("val_top5_accuracy", []), label="Val Top-5")
    plt.title(f"Fine-tuning Accuracy - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"fine_tune_accuracy_{tag}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.get("loss", []), label="Train Loss")
    plt.plot(epochs_range, history.get("val_loss", []), label="Val Loss")
    plt.title(f"Fine-tuning Loss - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"fine_tune_loss_{tag}.png", dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    best = load_best_experiment()
    tag = best["tag"]
    batch_size = best.get("batch_size") or DEFAULT_BATCH_SIZE
    model_path = Path(best["model_path"])
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Best model file not found: {model_path}")

    print("\n" + "=" * 72)
    print("FINE-TUNING FINAL BEST MODEL")
    print("=" * 72)
    print(f"Base model : {model_path}")
    print(f"Tag        : {tag}")
    print(f"Batch size : {batch_size}")
    print(f"LR         : {FINE_TUNE_LR}")
    print(f"Epochs     : {FINE_TUNE_EPOCHS}")

    train_gen, val_gen = build_generators(batch_size)
    class_weights = compute_class_weights(train_gen)

    model = tf.keras.models.load_model(model_path, compile=False)
    trainable_count = unfreeze_last_layers(model, FINE_TUNE_LAST_LAYERS)
    print(f"[INFO] Unfrozen non-BatchNorm layers: {trainable_count}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_DIR / f"fine_tuned_best_{tag}.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history_obj = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    final_path = OUTPUT_DIR / f"fine_tuned_final_{tag}.keras"
    model.save(final_path)

    history = {
        key: [float(v) for v in values]
        for key, values in history_obj.history.items()
    }
    save_json(OUTPUT_DIR / f"fine_tune_history_{tag}.json", history)
    save_plots(history, tag)

    result = {
        "phase": "fine_tuned",
        "tag": f"fine_tuned_{tag}",
        "base_tag": tag,
        "base_model_path": str(model_path),
        "best_model_path": str(OUTPUT_DIR / f"fine_tuned_best_{tag}.keras"),
        "final_model_path": str(final_path),
        "learning_rate": FINE_TUNE_LR,
        "batch_size": batch_size,
        "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS,
        "epochs_trained": len(history.get("loss", [])),
        "best_val_accuracy": float(max(history.get("val_accuracy", [0.0]))),
        "best_val_top5_accuracy": float(max(history.get("val_top5_accuracy", [0.0]))),
        "best_val_loss": float(min(history.get("val_loss", [999.0]))),
    }
    save_json(OUTPUT_DIR / f"fine_tune_result_{tag}.json", result)

    print("\n[DONE] Fine-tuning complete.")
    print(f"Best fine-tuned model : {result['best_model_path']}")
    print(f"Final fine-tuned model: {result['final_model_path']}")
    print("Run evaluate.py again to compare the fine-tuned model end-to-end.")


if __name__ == "__main__":
    main()
