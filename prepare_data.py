import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/news.csv")

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"])

# Combine title and article text
df["content"] = df["title"] + " " + df["text"]

# Keep only the columns we need
df = df[["content", "label"]]

# Display information
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts())

# Separate input and target
X = df["content"]
y = df["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))