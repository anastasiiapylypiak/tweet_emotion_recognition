# src/pipeline.py
# small script to demo single-sentence inference

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from preprocess import clean_text
from labelmap import EMOTIONS

MODEL_DIR = "models/roberta_emotion_en"

def load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

def predict_single(text, tokenizer, model):
    txt = clean_text(text)
    enc = tokenizer(txt, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    return {"text": txt, "emotion": EMOTIONS[idx], "confidence": float(probs[idx])}

if __name__ == "__main__":
    tok, mdl = load()
    examples = [
        "I am so happy about my new job!",
        "This is the worst day ever, I hate it.",
        "Well, that was surprising..."
    ]
    for ex in examples:
        print(predict_single(ex, tok, mdl))