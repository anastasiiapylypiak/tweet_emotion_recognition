# app.py (ENGLISH + PRETTY UI + ALWAYS-ON PLAYGROUND + NO DUPLICATES)

import os
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from labelmap import EMOTIONS  # list[str] mapping index -> emotion label

# ----------------------------
# CONFIG
# ----------------------------
MODEL_DIR = "models/roberta_emotion_en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEADERS = {
    # Explicit UA helps reduce 429/403
    "User-Agent": "tweet-emotion-recognition/1.0 (streamlit demo)"
}

# ----------------------------
# UI THEME / STYLING
# ----------------------------
def inject_css():
    st.markdown(
        """
        <style>
          /* Background */
          .stApp {
            background: radial-gradient(1200px 600px at 10% 0%, rgba(124,58,237,0.12), transparent 60%),
                        radial-gradient(900px 500px at 95% 15%, rgba(34,197,94,0.10), transparent 55%),
                        linear-gradient(180deg, rgba(15,23,42,0.02), rgba(15,23,42,0.00));
          }

          h1, h2, h3 { letter-spacing: -0.02em; }

          .card {
            padding: 16px 18px;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(255, 255, 255, 0.62);
            box-shadow: 0 8px 30px rgba(2, 6, 23, 0.08);
            backdrop-filter: blur(8px);
          }

          div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
          }

          .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.35);
            background: rgba(255,255,255,0.55);
            font-size: 12px;
            margin-right: 6px;
          }

          .muted { color: rgba(30,41,59,0.75); font-size: 13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# MODEL
# ----------------------------
@st.cache_resource
def load_model():
    # Optional: show a nicer error if the folder isn't there
    if os.path.isdir(MODEL_DIR):
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    else:
        # Fallback if demo machine doesn't have the local folder
        # (change/remove if you don't want any internet downloads)
        fallback = "arpanghoshal/EmoRoBERTa"
        tok = AutoTokenizer.from_pretrained(fallback)
        mdl = AutoModelForSequenceClassification.from_pretrained(fallback)

    mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

# ----------------------------
# REDDIT FETCH
# ----------------------------
@st.cache_data(ttl=60)
def fetch_posts_json(subreddit: str, sort: str = "new", limit: int = 25):
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": limit}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()

    data = r.json()
    children = data["data"]["children"]

    rows = []
    for c in children:
        d = c["data"]
        title = d.get("title", "") or ""
        selftext = d.get("selftext", "") or ""
        combined = (title + "\n\n" + selftext).strip()
        rows.append({
            "title": title,
            "selftext": selftext,
            "combined_text": combined,
            "permalink": "https://www.reddit.com" + d.get("permalink", ""),
            "score": d.get("score", None),
            "num_comments": d.get("num_comments", None),
            "upvote_ratio": d.get("upvote_ratio", None),
            "created_utc": d.get("created_utc", None),
        })
    return pd.DataFrame(rows)

# ----------------------------
# INFERENCE
# ----------------------------
def classify_with_probs(texts, tok, mdl, batch_size=16, max_len=192):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

    probs = np.vstack(all_probs) if all_probs else np.zeros((0, len(EMOTIONS)))
    pred_idx = probs.argmax(axis=1).tolist()
    conf = probs.max(axis=1).tolist()
    return pred_idx, conf, probs

def topk_labels(probs_row, k=3):
    idx = np.argsort(-probs_row)[:k]
    return [(EMOTIONS[i], float(probs_row[i])) for i in idx]

# ----------------------------
# PRETTY HELPERS
# ----------------------------
def style_confidence(df: pd.DataFrame):
    return (
        df.style
          .format({"confidence": "{:.0%}", "score": "{:,.0f}", "num_comments": "{:,.0f}", "upvote_ratio": "{:.2f}"})
          .background_gradient(subset=["confidence"], cmap="RdYlGn")
          .set_properties(**{"font-size": "13px"})
    )

def make_summary_cards(df: pd.DataFrame):
    avg_conf = float(df["confidence"].mean()) if not df.empty else 0.0
    top_emotion = df["pred_emotion"].value_counts().idxmax() if not df.empty else "—"
    top_emotion_count = int(df["pred_emotion"].value_counts().max()) if not df.empty else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Posts classified", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Average confidence", f"{avg_conf:.0%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Most frequent emotion", f"{top_emotion}", f"{top_emotion_count} posts")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# APP
# ----------------------------
def main():
    st.set_page_config(page_title="Reddit Emotion Lens", page_icon="🧠", layout="wide")
    inject_css()

    st.title("🧠 Reddit Emotion Lens")
    st.caption("Fetch recent Reddit posts and classify emotions using a fine-tuned RoBERTa model (Top-K makes it feel much more human).")

    tok, mdl = load_model()

    # Session state (store last results)
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "last_subreddit" not in st.session_state:
        st.session_state["last_subreddit"] = None

    # Sidebar
    with st.sidebar:
        st.subheader("Controls")

        subreddit = st.text_input(
            "Subreddit (without r/)",
            value=st.session_state["last_subreddit"] or "TrueOffMyChest",
            key="subreddit_input"
        )
        sort = st.selectbox("Sort", ["new", "hot", "top"], index=0, key="sort_select")
        limit = st.slider("Number of posts", 5, 100, 25, 5, key="limit_slider")

        use_body = st.toggle("Use title + body (recommended)", value=True, key="use_body_toggle")
        max_len = st.slider("Max tokens (context)", 64, 384, 192, 32, key="maxlen_slider")
        topk = st.slider("Show Top-K emotions", 1, 5, 3, 1, key="topk_slider")
        conf_floor = st.slider("Min confidence filter", 0.0, 1.0, 0.0, 0.05, key="conf_floor_slider")

        st.divider()
        st.markdown("**Mixed-emotion subreddits for demos:**")
        st.markdown("- `TrueOffMyChest`\n- `relationships`\n- `AmItheAsshole`\n- `confession`")
        st.markdown('<span class="pill">Red → low confidence</span><span class="pill">Green → high confidence</span>', unsafe_allow_html=True)

    # Tabs
    tab_overview, tab_posts, tab_playground = st.tabs(["📊 Overview", "🧾 Posts", "🧪 Playground"])

    # ----------------------------
    # ALWAYS-ON PLAYGROUND
    # ----------------------------
    with tab_playground:
        st.markdown("### Try your own text")
        st.markdown('<div class="muted">Great for live demos: paste something emotional and show Top-K output.</div>', unsafe_allow_html=True)

        sample = "I finally got the job offer… but I'm terrified I won't be good enough."
        user_text = st.text_area("Text to analyze", value=sample, height=140, key="playground_text")

        colA, colB = st.columns([1, 1])
        with colA:
            analyze = st.button("🔎 Analyze text", key="analyze_btn", use_container_width=True)
        with colB:
            st.caption(f"Device: **{DEVICE.type.upper()}** • Model: **{MODEL_DIR}**")

        if analyze:
            if not user_text.strip():
                st.warning("Please enter some text.")
            else:
                pred_idx, conf, probs = classify_with_probs([user_text], tok, mdl, batch_size=1, max_len=int(max_len))
                pred = EMOTIONS[pred_idx[0]]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"Prediction: **{pred}**")
                st.write(f"Confidence: **{conf[0]:.0%}**")
                st.markdown('</div>', unsafe_allow_html=True)

                tk = topk_labels(probs[0], k=int(topk))
                tk_df = pd.DataFrame(tk, columns=["emotion", "probability"])
                st.dataframe(
                    tk_df.style.format({"probability": "{:.0%}"})
                              .background_gradient(subset=["probability"], cmap="RdYlGn"),
                    use_container_width=True
                )

    # ----------------------------
    # FETCH + CLASSIFY
    # ----------------------------
    st.markdown("#### Reddit batch")
    fetch = st.button("🚀 Fetch + Classify", use_container_width=True, key="fetch_btn")

    if fetch:
        if not subreddit.strip():
            st.warning("Please enter a valid subreddit.")
        else:
            try:
                with st.spinner("Fetching Reddit posts…"):
                    df = fetch_posts_json(subreddit.strip(), sort=sort, limit=int(limit))
                    time.sleep(0.2)
            except requests.HTTPError as e:
                st.error(f"Reddit HTTP error: {e}")
                st.info("Try a different subreddit or wait a bit (rate limits happen).")
                df = None
            except Exception as e:
                st.error(f"Error: {e}")
                df = None

            if df is not None and not df.empty:
                texts = df["combined_text"].tolist() if use_body else df["title"].tolist()

                with st.spinner("Classifying emotions…"):
                    pred_idx, conf, probs = classify_with_probs(texts, tok, mdl, batch_size=16, max_len=int(max_len))

                df["pred_emotion"] = [EMOTIONS[i] for i in pred_idx]
                df["confidence"] = conf
                df["topk"] = [topk_labels(probs[i], k=int(topk)) for i in range(len(df))]

                if conf_floor > 0:
                    df = df[df["confidence"] >= conf_floor].reset_index(drop=True)

                st.session_state["results_df"] = df
                st.session_state["last_subreddit"] = subreddit.strip()

    df_res = st.session_state["results_df"]

    # ----------------------------
    # OVERVIEW TAB
    # ----------------------------
    with tab_overview:
        if df_res is None or df_res.empty:
            st.info("No results yet. Click **Fetch + Classify** to populate the dashboard.")
        else:
            make_summary_cards(df_res)

            st.markdown("### Emotion distribution")
            counts = df_res["pred_emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
            st.bar_chart(counts)

            st.markdown("### Confidence distribution")
            st.bar_chart(pd.Series(df_res["confidence"]).round(2).value_counts().sort_index())

    # ----------------------------
    # POSTS TAB
    # ----------------------------
    with tab_posts:
        if df_res is None or df_res.empty:
            st.info("No results yet. Click **Fetch + Classify** to populate the table.")
        else:
            st.markdown("### Results table")
            show_cols = ["title", "pred_emotion", "confidence", "score", "num_comments", "upvote_ratio", "permalink"]
            df_show = df_res[show_cols].copy()

            st.dataframe(style_confidence(df_show), use_container_width=True, height=420)

            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results (CSV)", csv, file_name="reddit_emotions.csv", mime="text/csv")

            st.markdown("### Details (Top-K emotions per post)")
            for _, row in df_res.iterrows():
                title = row["title"][:140] + ("…" if len(row["title"]) > 140 else "")
                with st.expander(f"{row['pred_emotion']} • {row['confidence']:.0%} — {title}"):
                    st.write(f"**Link:** {row['permalink']}")
                    if use_body and row.get("selftext"):
                        st.write(row["selftext"][:2000] + ("…" if len(row["selftext"]) > 2000 else ""))

                    tk_df = pd.DataFrame(row["topk"], columns=["emotion", "probability"])
                    st.dataframe(
                        tk_df.style.format({"probability": "{:.0%}"})
                                  .background_gradient(subset=["probability"], cmap="RdYlGn"),
                        use_container_width=True,
                        height=150
                    )
                    st.caption("Tip: in real text, the “right” emotion is often in Top-3 even if Top-1 is debatable.")

# ----------------------------
# ENTRYPOINT
# ----------------------------
if __name__ == "__main__":
    main()
