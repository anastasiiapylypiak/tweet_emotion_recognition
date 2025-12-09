from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from labelmap import EMOTIONS  # ← your 7 buckets

model_path = "models/roberta_emotion_en"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = {i: e for i, e in enumerate(EMOTIONS)}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    pred_id = int(torch.argmax(probs))
    return id2label[pred_id], float(probs[0][pred_id])

if __name__ == "__main__":
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break
        label, confidence = predict(text)
        print(f"Prediction: {label} ({confidence:.2f})")