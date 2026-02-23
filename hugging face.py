# =====================================
# HUGGING FACE — PRETRAINED MODEL
# =====================================

# -------------------------------
# Step 1: Upload Dataset
# -------------------------------
from google.colab import files
uploaded = files.upload()

# -------------------------------
# Step 2: Import Libraries
# -------------------------------
import pandas as pd
from transformers import pipeline

# -------------------------------
# Step 3: Load Dataset
# -------------------------------
df = pd.read_csv(list(uploaded.keys())[0])

# -------------------------------
# Step 4: Load Pretrained Model
# -------------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# -------------------------------
# Step 5: Select Text Column
# -------------------------------
# Change column name if needed
text_column = df.columns[0]   # using first column as text
texts = df[text_column].astype(str).tolist()

# -------------------------------
# Step 6: Run Predictions
# -------------------------------
results = sentiment_analyzer(texts)

# -------------------------------
# Step 7: Add Predictions to Dataset
# -------------------------------
df["Prediction"] = [r["label"] for r in results]
df["Confidence"] = [r["score"] for r in results]

print("\n=== Dataset with Hugging Face Predictions ===\n")
print(df.head())

# -------------------------------
# Step 8: Display Sample Results
# -------------------------------
print("\n=== Sample Predictions ===\n")

for text, result in zip(texts[:10], results[:10]):
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} (Confidence: {result['score']:.2f})\n")

Kaggle link-https://www.kaggle.com/datasets/mabubakrsiddiq/student-exam-performance
