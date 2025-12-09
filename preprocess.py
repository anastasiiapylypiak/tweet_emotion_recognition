# src/preprocess.py
# Download GoEmotions, clean text, map labels to 7 classes and save processed files.

import re
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from labelmap import map_label_name_to_bucket, bucket_name_to_index

URL_RE = re.compile(r"http\S+")
MENTION_RE = re.compile(r"@\w+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def extract_label_from_row(row, id2label):
    """
    GoEmotions rows contain 'labels' as a list/array of label ids (multi-label).
    For simplicity, pick the first label if present, map to bucket.
    If no labels, return 'neutral'.
    """
    label_ids = row.get("labels", [])

    # label_ids may be numpy array → use len()
    if len(label_ids) == 0:
        mapped = "neutral"
    else:
        first_id = int(label_ids[0])
        label_name = id2label[first_id]
        mapped = map_label_name_to_bucket(label_name)

    return bucket_name_to_index(mapped)

def process_and_save():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("go_emotions")
    id2label = ds["train"].features["labels"].feature.int2str  # function mapping id->name
    # alternatively build a dict:
    id2label_map = {i: id2label(i) for i in range(len(ds["train"].features["labels"].feature.names))}
    # Process each split
    for split in ["train", "validation", "test"]:
        print(f"Processing split: {split}")
        df = ds[split].to_pandas()
        df["text_clean"] = df["text"].apply(clean_text)
        # map to single int label 0..6
        df["label"] = df.apply(lambda r: extract_label_from_row(r, id2label_map), axis=1)
        # keep only useful columns
        out = df[["text_clean", "label"]].rename(columns={"text_clean": "text"})
        out.to_parquet(f"data/processed/goemotions_{split}_7labels.parquet", index=False)
        print(f"Saved data/processed/goemotions_{split}_7labels.parquet rows={len(out)}")

if __name__ == "__main__":
    process_and_save()