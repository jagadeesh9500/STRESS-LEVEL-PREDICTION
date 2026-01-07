"""
Configuration settings for Stress Level Prediction System
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
STRESS_MODEL_PATH = os.path.join(MODELS_DIR, "stress_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pt")

# Upload settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Database
DATABASE_PATH = os.path.join(BASE_DIR, "stress_history.db")

# Emotion labels (must match training order)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Map emotions to stress levels (1-10 scale)
EMOTION_STRESS_MAP = {
    'angry': 8,
    'disgust': 6,
    'fear': 9,
    'happy': 2,
    'neutral': 4,
    'sad': 7,
    'surprise': 3
}

# Map emotions to stress categories
EMOTION_STRESS_CATEGORY = {
    'angry': ('High Stress', 3),
    'disgust': ('Moderate Stress', 2),
    'fear': ('High Stress', 3),
    'happy': ('Low Stress', 1),
    'neutral': ('Moderate Stress', 2),
    'sad': ('High Stress', 3),
    'surprise': ('Low Stress', 1)
}

# Text sentiment thresholds
TEXT_STRESS_THRESHOLDS = {
    'high': -0.3,      # polarity < -0.3 = high stress
    'moderate': 0.1,   # polarity < 0.1 = moderate stress
    'low': 1.0         # polarity >= 0.1 = low stress
}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
