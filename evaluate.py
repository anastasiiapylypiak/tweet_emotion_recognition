# evaluate.py
# Evaluate trained model on test split and write metrics + confusion matrix + detailed CSV + extra visualizations

import os
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

from labelmap import EMOTIONS  # liste des 7 émotions dans le bon ordre

# 🔧 À adapter si besoin : chemin du modèle et du test set
MODEL_DIR = "models/roberta_emotion_en"
TEST_PARQUET = "data/processed/goemotions_test_7labels.parquet"

RESULTS_METRICS_DIR = Path("results/metrics")
RESULTS_FIGURES_DIR = Path("results/figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer():
    """
    Charge le tokenizer et le modèle fine-tuné.
    """
    print(f"🔹 Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model, batch_size=32):
    """
    Fait des prédictions par batch.
    Retourne:
      - preds: labels prédits (numpy array)
      - confidences: score de confiance (proba de la classe prédite)
    """
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
            outputs = model(**enc)
            logits = outputs.logits
            probs = softmax(logits, dim=-1)  # [batch, num_labels]

        batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
        batch_confidences = probs.max(dim=-1).values.cpu().numpy()

        all_preds.extend(batch_preds.tolist())
        all_confidences.extend(batch_confidences.tolist())

    return np.array(all_preds), np.array(all_confidences)


def plot_confusion_matrix(cm, labels, filename, normalize=False):
    """
    Trace et sauvegarde une matrice de confusion.
    """
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
    out_path = RESULTS_FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved confusion matrix to {out_path}")


def plot_f1_per_class(f1_per_class, labels, filename):
    """
    Barplot des F1 par classe (permet de voir quelles émotions sont bien apprises ou non).
    """
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, f1_per_class)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.0)
    plt.title("F1-score per emotion (test set)")
    plt.tight_layout()
    out_path = RESULTS_FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved F1 per class barplot to {out_path}")


def plot_label_distribution(y_true, y_pred, labels, filename):
    """
    Compare la distribution des labels vrais vs prédits.
    Utile pour voir si le modèle sur-prédit certaines émotions.
    """
    true_counts = np.bincount(y_true, minlength=len(labels))
    pred_counts = np.bincount(y_pred, minlength=len(labels))

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts, width, label="True")
    plt.bar(x + width / 2, pred_counts, width, label="Predicted")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution: true vs predicted (test set)")
    plt.legend()
    plt.tight_layout()
    out_path = RESULTS_FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved label distribution plot to {out_path}")


def plot_confidence_histogram(confidences, correct, filename):
    """
    Histogramme des confiances pour les prédictions correctes vs incorrectes.
    Permet de voir si le modèle est bien calibré ou trop confiant à tort.
    """
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
    out_path = RESULTS_FIGURES_DIR / filename
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved confidence histogram to {out_path}")


def main():
    # Crée les dossiers de sortie
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger le test set et le modèle
    print(f"🔹 Loading test set from: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)

    tokenizer, model = load_model_and_tokenizer()

    texts = df["text"].tolist()
    y_true = df["label"].to_numpy()

    # 2. Prédictions + confiance
    print("🔹 Predicting on test set...")
    y_pred, confidences = predict_batch(texts, tokenizer, model)

    # 3. Métriques globales
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    print("\n=== Global metrics on test set ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")
    print("\nF1 per class:")
    for label, f1_val in zip(EMOTIONS, f1_per_class):
        print(f"  {label:8s} : {f1_val:.4f}")

    # Sauvegarde metrics globales
    metrics_txt = RESULTS_METRICS_DIR / "test_metrics.txt"
    with metrics_txt.open("w", encoding="utf-8") as f:
        f.write("Global metrics on test set\n")
        f.write(f"Accuracy    : {acc:.4f}\n")
        f.write(f"Macro F1    : {macro_f1:.4f}\n")
        f.write(f"Weighted F1 : {weighted_f1:.4f}\n\n")
        f.write("F1 per class:\n")
        for label, f1_val in zip(EMOTIONS, f1_per_class):
            f.write(f"{label:8s} : {f1_val:.4f}\n")
    print(f"\n💾 Saved global metrics to {metrics_txt}")

    # 4. Classification report complet
    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTIONS,
        digits=4,
    )
    print("\n=== Classification report ===")
    print(report)

    report_txt = RESULTS_METRICS_DIR / "test_classification_report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"💾 Saved classification report to {report_txt}")

    # 5. Matrices de confusion
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    plot_confusion_matrix(cm, EMOTIONS, "confusion_matrix_test_raw.png", normalize=False)
    plot_confusion_matrix(cm, EMOTIONS, "confusion_matrix_test_normalized.png", normalize=True)

    # 6. Visualisation F1 par classe
    plot_f1_per_class(f1_per_class, EMOTIONS, "f1_per_class_barplot.png")

    # 7. Distribution des labels vrais vs prédits
    plot_label_distribution(y_true, y_pred, EMOTIONS, "label_distribution_true_vs_pred.png")

    # 8. CSV détaillé pour l’error analysis
    print("🔹 Saving detailed predictions with confidence for error analysis...")

    true_emotions = [EMOTIONS[i] for i in y_true]
    pred_emotions = [EMOTIONS[i] for i in y_pred]
    correct = (y_true == y_pred)

    detailed_df = pd.DataFrame(
        {
            "text": texts,
            "true_label": y_true,
            "true_emotion": true_emotions,
            "pred_label": y_pred,
            "pred_emotion": pred_emotions,
            "correct": correct,
            "confidence": confidences,
        }
    )

    detailed_csv = RESULTS_METRICS_DIR / "test_predictions_with_confidence.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"💾 Saved detailed predictions to {detailed_csv}")

    # 9. Histogramme des confiances
    plot_confidence_histogram(confidences, correct, "confidence_hist_correct_vs_wrong.png")

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
