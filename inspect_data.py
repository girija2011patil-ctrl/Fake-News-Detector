import pandas as pd

file_path = "dataset/news.csv"

df = pd.read_csv(file_path)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nUnique labels:")
print(df["label"].unique())