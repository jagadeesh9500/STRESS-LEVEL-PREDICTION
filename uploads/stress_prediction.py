import joblib
import pandas as pd
from advice_engine import generate_advice

# Load fresh model
model = joblib.load("models/stress_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature names MUST match dataset
FEATURES = [
    'anxiety_level',
    'self_esteem',
    'mental_health_history',
    'depression',
    'headache',
    'blood_pressure',
    'sleep_quality',
    'breathing_problem',
    'noise_level',
    'living_conditions',
    'safety',
    'basic_needs',
    'academic_performance',
    'study_load',
    'teacher_student_relationship',
    'future_career_concerns',
    'social_support',
    'peer_pressure',
    'extracurricular_activities',
    'bullying'
]


tests = {
    "LOW": [1]*20,
    "MEDIUM": [5]*20,
    "HIGH": [10]*20
}

for name, values in tests.items():
    df = pd.DataFrame([values], columns=FEATURES)
    df_scaled = scaler.transform(df)

    raw_pred = model.predict(df_scaled)[0]

    # Normalize classes
    if raw_pred == -1:
        pred = 0
    elif raw_pred == 0:
        pred = 1
    else:
        pred = 2

    print(f"\n{name} INPUT → Raw class:", raw_pred)
    print(f"{name} INPUT → Normalized class:", pred)

    for tip in generate_advice(pred):
        print("-", tip)

