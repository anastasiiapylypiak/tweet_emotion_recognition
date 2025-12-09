from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your trained model
model_path = "models/roberta_emotion_en"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Emotion index to label mapping (GoEmotions 28)
id2label = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)
    pred_id = int(torch.argmax(probs))
    return id2label[pred_id], float(probs[0][pred_id])


# Try it
if __name__ == "__main__":
    while True:
        text = input("\nEnter text: ")
        if text.lower() == "exit":
            break
        label, confidence = predict(text)
        print(f"Prediction: {label} ({confidence:.2f})")