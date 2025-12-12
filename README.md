# 🧠 Reddit Emotion Lens  
**Emotion detection on real Reddit posts using a fine-tuned RoBERTa model**

---

## 📌 Project overview

**Reddit Emotion Lens** is a Streamlit web application that analyzes the emotional content of Reddit posts.

The app:
- fetches recent posts from any subreddit,
- classifies each post into **human emotions** using a **RoBERTa model fine-tuned on GoEmotions**,
- displays results with **confidence scores**, **Top-K emotions**, and **visual analytics**.

The goal is **exploratory and demonstrative**:
- show how NLP emotion detection behaves on real-world, noisy text,
- highlight mixed emotions rather than a single “perfect” label,
- provide an interactive demo for analysis and discussion.

---

## ✨ Key features

- 🔎 **Reddit scraping** (no API key required)
- 🧠 **Emotion classification** (RoBERTa, fine-tuned)
- 📊 **Emotion distribution & confidence visualization**
- 🎨 **Confidence color gradient** (red → green)
- 🧪 **Playground** to test any custom text live
- 🏆 **Top-K emotions per post** (much more realistic than Top-1 only)
- 💾 **CSV export** of results

---

## 🧩 Model

- Architecture: **RoBERTa**
- Task: multi-class emotion classification
- Labels: based on **GoEmotions**
- Inference only (no training in the app)
- Uses CPU by default, GPU automatically if available

---

## 🚀 Getting started (for collaborators)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/anastasiiapylypiak/tweet_emotion_recognition.git
cd tweet_emotion_recognition
```

### 2️⃣ Clone the repository

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
If requirements.txt is missing, install manually:
```bash
pip install streamlit torch transformers pandas numpy requests
```

### 4️⃣ Run the application
⚠️ Important: always run Streamlit via Python to ensure the correct environment is used.
```bash
python -m streamlit run app.py
```
The app will open automatically in your browser at:
👉 http://localhost:8501

## 🧭 How to use the app

### 🔹 Playground tab (recommended for demos)

The Playground allows you to test the emotion classifier on any custom text.

**Steps:**
1. Paste any text (e.g. a Reddit post, a tweet, or a personal message)
2. Click **Analyze**
3. Observe the results:
   - **Predicted emotion**
   - **Confidence score**
   - **Top-K emotions** (especially useful for mixed or ambiguous feelings)

This tab is ideal for live demonstrations and qualitative analysis.

---

### 🔹 Reddit batch analysis

This mode lets you analyze emotions across multiple real Reddit posts.

**Steps:**
1. Choose a subreddit (without `r/`)
2. Click **Fetch + Classify**
3. Explore the outputs:
   - **Emotion distribution** across posts
   - **Confidence levels** of predictions
   - **Individual post details** with Top-K emotions
4. Download the results as a **CSV file** if needed

This mode is useful for observing global emotional trends and testing the model on real-world data.
