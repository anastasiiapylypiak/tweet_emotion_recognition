# src/evaluate.py
# Evaluate trained model on test split and write metrics + confusion matrix

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from labelmap import EMOTIONS

MODEL_DIR = "models/roberta_emotion_en"
TEST_PARQUET = "data/processed/goemotions_test_7labels.parquet"

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict_batch(texts, tokenizer, model, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        batch_preds = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(batch_preds)
    return preds

def main():
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TEST_PARQUET)
    tokenizer, model = load_model_and_tokenizer()

    texts = df["text"].tolist()
    y_true = df["label"].tolist()
    y_pred = predict_batch(texts, tokenizer, model)

    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    print(report)
    with open("results/metrics/test_classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    # plot confusion matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = range(len(EMOTIONS))
    plt.xticks(tick_marks, EMOTIONS, rotation=45, ha='right')
    plt.yticks(tick_marks, EMOTIONS)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix_test.png")
    print("Saved confusion matrix to results/figures/confusion_matrix_test.png")

if __name__ == "__main__":
    main()