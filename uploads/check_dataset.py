import pandas as pd

df = pd.read_csv("dataset/StressLevelDataset.csv")

print("Columns in dataset:")
print(df.columns)

print("\nStress level distribution:")
print(df["stress_level"].value_counts())
