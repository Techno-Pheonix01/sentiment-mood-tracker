from transformers import pipeline
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # force CPU
)

print("---- Sentiment Analyzer ----")
print("Type 'exit' to stop.\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        print("Bye 👋")
        break

    result = sentiment_model(text)[0]
    label = result["label"]
    score = result["score"]

    if score < 0.75:
        label = "NEUTRAL"

    print(f"✅ Sentiment: {label}")
    print(f"📌 Confidence: {round(score*100, 2)} %\n")
