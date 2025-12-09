# src/train_classifier.py
# Fine-tune roberta-base on processed GoEmotions 7-label dataset using HuggingFace Trainer (CPU-only)

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from labelmap import EMOTIONS

MODEL_NAME = "roberta-base"
MAX_LEN = 128
OUTPUT_DIR = "models/roberta_emotion_en"


def load_dataset_from_parquet(path):
    df = pd.read_parquet(path)
    return Dataset.from_pandas(df)


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


# -------------------------------
# Weighted Trainer for CPU
# -------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # accepts extra kwargs to avoid errors
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,  # compute loss manually
        )
        logits = outputs.logits

        # Move class weights to same device as logits
        cw = self.class_weights.to(logits.device)
        loss_fct = CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    # ensure output folder exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(EMOTIONS)
    )

    # -------------------------------
    # Load datasets
    # -------------------------------
    train_ds = load_dataset_from_parquet(
        "data/processed/goemotions_train_7labels.parquet"
    )
    val_ds = load_dataset_from_parquet(
        "data/processed/goemotions_validation_7labels.parquet"
    )

    # Tokenize
    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    val_ds = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    # remove unused column
    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    # rename label column
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    # correct PyTorch formatting
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------------
    # Compute class weights
    # -------------------------------
    labels = torch.tensor(train_ds["labels"])
    num_classes = len(EMOTIONS)
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize

    print("\nClass weights used:")
    for lbl, w in zip(EMOTIONS, class_weights):
        print(f"{lbl}: {w:.4f}")

    # -------------------------------
    # Training arguments (CPU)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=200,
        fp16=False,  # CPU only
    )

    # -------------------------------
    # Initialize trainer
    # -------------------------------
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Train
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)

    print("Training complete. Model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()