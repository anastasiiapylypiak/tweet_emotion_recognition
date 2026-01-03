# evaluate.py
# Full evaluation: metrics, Top-K accuracy, calibration, error analysis, plots

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

from labelmap import EMOTIONS

# ----------------------------
# CONFIG
# ----------------------------
MODEL_DIR = "models/roberta_emotion_en"
TEST_PARQUET = "data/processed/goemotions_test_7labels.parquet"

RESULTS_METRICS_DIR = Path("results/metrics")
RESULTS_FIGURES_DIR = Path("results/figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model_and_tokenizer():
    print(f"🔹 Loading model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ----------------------------
# INFERENCE
# ----------------------------
def predict_batch(texts, tokenizer, model, batch_size=32):
    all_preds = []
    all_confidences = []
    all_probs = []

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

        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        confs = probs.max(dim=-1).values.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_confidences.extend(confs.tolist())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_confidences),
        np.vstack(all_probs),
    )


# ----------------------------
# METRICS
# ----------------------------
def topk_accuracy(y_true, probs, k=3):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])


# ----------------------------
# PLOTS
# ----------------------------
def plot_confusion_matrix(cm, labels, filename, normalize=False):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix" + (" (normalized)" if normalize else ""))

    out = RESULTS_FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


def plot_f1_per_class(f1_vals, labels, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, f1_vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title("F1-score per emotion")
    plt.tight_layout()

    out = RESULTS_FIGURES_DIR / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


def plot_label_distribution(y_true, y_pred, labels, filename):
    true_counts = np.bincount(y_true, minlength=len(labels))
    pred_counts = np.bincount(y_pred, minlength=len(labels))

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, true_counts, width, label="True")
    plt.bar(x + width / 2, pred_counts, width, label="Predicted")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Label distribution (test set)")
    plt.legend()
    plt.tight_layout()

    out = RESULTS_FIGURES_DIR / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


def plot_confidence_histogram(confidences, correct, filename):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 20)
    plt.hist(confidences[correct], bins=bins, alpha=0.6, label="Correct")
    plt.hist(confidences[~correct], bins=bins, alpha=0.6, label="Incorrect")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution")
    plt.legend()
    plt.tight_layout()

    out = RESULTS_FIGURES_DIR / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


def plot_reliability_diagram(confidences, correct, filename, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(confidences, edges) - 1

    accs, confs = [], []
    for i in range(bins):
        mask = bin_ids == i
        accs.append(correct[mask].mean() if mask.any() else 0)
        confs.append(confidences[mask].mean() if mask.any() else 0)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(confs, accs, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = RESULTS_FIGURES_DIR / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


def plot_confidence_per_emotion(confidences, y_pred, labels, filename):
    df = pd.DataFrame(
        {
            "emotion": [labels[i] for i in y_pred],
            "confidence": confidences,
        }
    )

    plt.figure(figsize=(8, 5))
    df.boxplot(by="emotion", column="confidence", grid=False)
    plt.suptitle("")
    plt.title("Confidence per predicted emotion")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out = RESULTS_FIGURES_DIR / filename
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved {out}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Loading test set: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)

    tokenizer, model = load_model_and_tokenizer()

    texts = df["text"].tolist()
    y_true = df["label"].to_numpy()

    print("🔹 Running inference...")
    y_pred, confidences, probs = predict_batch(texts, tokenizer, model)
    correct = y_true == y_pred

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None)

    top3 = topk_accuracy(y_true, probs, k=3)
    top5 = topk_accuracy(y_true, probs, k=5)

    # Save metrics
    with (RESULTS_METRICS_DIR / "test_metrics.txt").open("w") as f:
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {weighted_f1:.4f}\n")
        f.write(f"Top-3 Acc     : {top3:.4f}\n")
        f.write(f"Top-5 Acc     : {top5:.4f}\n\n")
        for lbl, f1 in zip(EMOTIONS, f1_per_class):
            f.write(f"{lbl:8s}: {f1:.4f}\n")

    # Report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    (RESULTS_METRICS_DIR / "classification_report.txt").write_text(report)

    # Plots
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, EMOTIONS, "confusion_raw.png")
    plot_confusion_matrix(cm, EMOTIONS, "confusion_normalized.png", normalize=True)
    plot_f1_per_class(f1_per_class, EMOTIONS, "f1_per_class.png")
    plot_label_distribution(y_true, y_pred, EMOTIONS, "label_distribution.png")
    plot_confidence_histogram(confidences, correct, "confidence_histogram.png")
    plot_reliability_diagram(confidences, correct, "reliability_diagram.png")
    plot_confidence_per_emotion(confidences, y_pred, EMOTIONS, "confidence_per_emotion.png")

    # Error analysis
    detailed = pd.DataFrame(
        {
            "text": texts,
            "true_emotion": [EMOTIONS[i] for i in y_true],
            "pred_emotion": [EMOTIONS[i] for i in y_pred],
            "confidence": confidences,
            "correct": correct,
        }
    )

    detailed["error_type"] = detailed.apply(
        lambda r: "correct" if r.correct else f"{r.true_emotion} → {r.pred_emotion}",
        axis=1,
    )

    detailed.to_csv(RESULTS_METRICS_DIR / "predictions_detailed.csv", index=False)

    (
        detailed[~detailed.correct]
        .groupby("error_type")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_csv(RESULTS_METRICS_DIR / "top_confusions.csv")
    )

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()