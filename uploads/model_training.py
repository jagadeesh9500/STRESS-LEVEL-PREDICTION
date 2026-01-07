import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("dataset/StressLevelDataset.csv")

# Features and target
X = df.drop("stress_level", axis=1)
y = df["stress_level"]   # 1 = Low, 2 = Moderate, 3 = High

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# Evaluation
print("\nMODEL EVALUATION\n")
print(classification_report(y_test, model.predict(X_test_scaled)))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stress_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Model trained and saved successfully")
