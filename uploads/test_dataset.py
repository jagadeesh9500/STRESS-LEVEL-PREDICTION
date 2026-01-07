import pandas as pd

df = pd.read_csv("dataset/StressLevelDataset.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nStress level distribution:")
print(df["stress_level"].value_counts())
