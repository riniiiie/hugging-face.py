# =====================================
# NAIVE BAYES — SUPERVISED LEARNING
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 3: Load Dataset
# -------------------------------
df = pd.read_csv(list(uploaded.keys())[0])

# -------------------------------
# Step 4: Data Preprocessing
# -------------------------------

# Drop timestamp if exists
if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])

# Convert categorical columns → numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Select target column (change if needed)
target_column = df_encoded.columns[-1]

X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]

# -------------------------------
# Step 5: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 6: Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 7: Train Naive Bayes Model
# -------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------
# Step 8: Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 9: Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Naive Bayes Results ===\n")
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

Kaggle link-https://www.kaggle.com/datasets/mabubakrsiddiq/student-exam-performance
