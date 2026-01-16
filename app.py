import streamlit as st
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # force CPU
)


st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("💬 Sentiment Analysis (SDG 3 Mini Project)")
st.write("Enter text and check if it’s Positive / Negative / Neutral.")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

text = st.text_area("Enter your text:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = sentiment_model(text)[0]
        label = result["label"]
        score = result["score"]

        # Neutral threshold
        if score < 0.75:
            label = "NEUTRAL"

        st.success(f"✅ Sentiment: {label}")
        st.info(f"📌 Confidence: {round(score * 100, 2)} %")

        if label == "NEGATIVE":
            st.warning("💛 Take a deep breath. If you're overwhelmed, talk to someone you trust.")
        elif label == "POSITIVE":
            st.balloons()
            st.write("🌟 Love this energy! Keep going!")
        else:
            st.write("🙂 Seems neutral. Want to share more?")
