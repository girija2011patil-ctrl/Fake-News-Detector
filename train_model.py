import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("dataset/news.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ==================================================
# 2. REMOVE UNNECESSARY COLUMN
# ==================================================

df = df.drop(columns=["Unnamed: 0"])


# ==================================================
# 3. COMBINE TITLE AND ARTICLE TEXT
# ==================================================

df["content"] = df["title"] + " " + df["text"]


# ==================================================
# 4. SELECT INPUT AND TARGET
# ==================================================

X = df["content"]
y = df["label"]


# ==================================================
# 5. CHECK LABEL DISTRIBUTION
# ==================================================

print("\nLabel distribution:")
print(y.value_counts())


# ==================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# 7. TF-IDF VECTORIZATION
# ==================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Learn vocabulary from training data
# and convert training text into numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

# Convert test text using the same vocabulary
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)


# ==================================================
# 8. CREATE MACHINE LEARNING MODEL
# ==================================================

model = LogisticRegression(
    max_iter=1000
)


# ==================================================
# 9. TRAIN MODEL
# ==================================================

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# ==================================================
# 10. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test_tfidf)


# ==================================================
# 11. CALCULATE ACCURACY
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", round(accuracy * 100, 2), "%")


# ==================================================
# 12. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==================================================
# 13. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==================================================
# 14. SAVE TRAINED MODEL
# ==================================================

joblib.dump(model, "model.pkl")

print("\nModel saved as: model.pkl")


# ==================================================
# 15. SAVE TF-IDF VECTORIZER
# ==================================================

joblib.dump(vectorizer, "vectorizer.pkl")

print("Vectorizer saved as: vectorizer.pkl")


# ==================================================
# 16. FINAL MESSAGE
# ==================================================

print("\n======================================")
print("FAKE NEWS DETECTOR MODEL READY!")
print("======================================")