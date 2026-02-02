from transformers import pipeline

# -------------------------------
# Step 1: Load Pretrained Model
# -------------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------------
# Step 2: Same Text Samples
# -------------------------------
texts = [
    "Congratulations you won a free gift",
    "Meeting scheduled tomorrow",
    "Claim your prize now",
    "Please review the attached document",
    "Urgent account verification required",
    "Let us discuss the project",
    "Limited offer buy now",
    "Your OTP is 563829",
    "Win cash rewards instantly",
    "Can we reschedule the meeting"
]

# -------------------------------
# Step 3: Run Sentiment Analysis
# -------------------------------
results = sentiment_analyzer(texts)

# -------------------------------
# Step 4: Display Predictions
# -------------------------------
print("\n=== Hugging Face Pretrained Model Predictions ===\n")

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} (Confidence: {result['score']:.2f})\n")
