# idioms_evaluation.py
# Evaluate idioms (German + French) on English emotion model

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from labelmap import EMOTIONS  # ["joy","anger","sadness","fear","love","surprise","neutral"]

MODEL_DIR = "models/roberta_emotion_en"

# Add French idioms CSV here
IDIOMS_CONFIGS = [
    {"name": "de", "csv": "data/german_idioms.csv"},
    {"name": "fr", "csv": "data/french_idioms.csv"},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    print(f"🔹 Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model



def predict_batch(texts, tokenizer, model, batch_size=32):
    all_preds = []
    all_confidences = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = softmax(logits, dim=-1)

        batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
        batch_confidences = probs.max(dim=-1).values.cpu().numpy()

        all_preds.extend(batch_preds.tolist())
        all_confidences.extend(batch_confidences.tolist())

    return np.array(all_preds), np.array(all_confidences)


def plot_confusion_matrix(cm, labels, out_path, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    title = "Confusion matrix (normalized)" if normalize else "Confusion matrix"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved confusion matrix to {out_path}")


def plot_f1_per_class(f1_per_class, labels, out_path):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, f1_per_class)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.0)
    plt.title("F1-score per emotion (idioms)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved F1 per class barplot to {out_path}")


def plot_label_distribution(y_true, y_pred, labels, out_path):
    true_counts = np.bincount(y_true, minlength=len(labels))
    pred_counts = np.bincount(y_pred, minlength=len(labels))

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts, width, label="True")
    plt.bar(x + width / 2, pred_counts, width, label="Predicted")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution: true vs predicted (idioms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved label distribution plot to {out_path}")


def plot_confidence_histogram(confidences, correct, out_path):
    conf_correct = confidences[correct]
    conf_wrong = confidences[~correct]

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 20)
    plt.hist(conf_correct, bins=bins, alpha=0.6, label="Correct")
    plt.hist(conf_wrong, bins=bins, alpha=0.6, label="Incorrect")
    plt.xlabel("Confidence (max softmax)")
    plt.ylabel("Number of samples")
    plt.title("Confidence distribution: correct vs incorrect predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved confidence histogram to {out_path}")


def evaluate_one_language(lang_name, idioms_csv, tokenizer, model):
    results_metrics_dir = Path(f"results/idioms_metrics_{lang_name}")
    results_figures_dir = Path(f"results/idioms_figures_{lang_name}")
    results_metrics_dir.mkdir(parents=True, exist_ok=True)
    results_figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==============================")
    print(f"🔹 Evaluating idioms: {lang_name.upper()}")
    print(f"🔹 Loading idioms from {idioms_csv}")

    df = pd.read_csv(idioms_csv)
    print(f"Columns in CSV: {list(df.columns)}")

    # Keep only idioms with emotions in EMOTIONS
    initial_count = len(df)
    df = df[df["true_emotion"].isin(EMOTIONS)].copy()
    skipped = initial_count - len(df)
    print(f"ℹ️ {skipped} rows skipped: emotion not in model label space")

    texts = df["text"].tolist()
    y_true = np.array([EMOTIONS.index(e) for e in df["true_emotion"]])

    # Predict
    print("🔹 Predicting on idioms...")
    y_pred, confidences = predict_batch(texts, tokenizer, model)
    correct = (y_true == y_pred)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(len(EMOTIONS))), zero_division=0
    )

    print("\n=== Idioms Evaluation ===")
    print(f"[{lang_name}] Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    print("F1 per class:")
    for label, f1_val in zip(EMOTIONS, f1_per_class):
        print(f"  {label:8s} : {f1_val:.4f}")

    # Save metrics
    metrics_txt = results_metrics_dir / "idioms_metrics.txt"
    with metrics_txt.open("w", encoding="utf-8") as f:
        f.write(f"Language: {lang_name}\n")
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n")
        f.write("F1 per class:\n")
        for label, f1_val in zip(EMOTIONS, f1_per_class):
            f.write(f"{label:8s} : {f1_val:.4f}\n")
    print(f"💾 Saved metrics to {metrics_txt}")

    # Classification report
    report = classification_report(
        y_true, y_pred, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS, digits=4, zero_division=0
    )
    report_txt = results_metrics_dir / "idioms_classification_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 Saved classification report to {report_txt}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    plot_confusion_matrix(cm, EMOTIONS, results_figures_dir / "confusion_matrix_raw.png", normalize=False)
    plot_confusion_matrix(cm, EMOTIONS, results_figures_dir / "confusion_matrix_normalized.png", normalize=True)

    # F1 per class plot
    plot_f1_per_class(f1_per_class, EMOTIONS, results_figures_dir / "f1_per_class.png")

    # Label distribution
    plot_label_distribution(y_true, y_pred, EMOTIONS, results_figures_dir / "label_distribution.png")

    # Confidence histogram
    plot_confidence_histogram(confidences, correct, results_figures_dir / "confidence_hist_correct_vs_wrong.png")

    # Save detailed CSV
    detailed_df = pd.DataFrame({
        "text": texts,
        "true_emotion": df["true_emotion"].values,
        "pred_label": y_pred,
        "pred_emotion": [EMOTIONS[i] for i in y_pred],
        "correct": correct,
        "confidence": confidences,
        "category": df["category"].values if "category" in df.columns else None,
    })
    detailed_csv = results_metrics_dir / "idioms_detailed_predictions.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"💾 Saved detailed predictions to {detailed_csv}")

    print(f"✅ {lang_name.upper()} idioms evaluation complete.")


def main():
    tokenizer, model = load_model_and_tokenizer()

    for cfg in IDIOMS_CONFIGS:
        evaluate_one_language(cfg["name"], cfg["csv"], tokenizer, model)


if __name__ == "__main__":
    main()
