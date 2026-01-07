from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import numpy as np
import joblib
import uuid
import os
import base64
from werkzeug.utils import secure_filename
from advice_engine import generate_advice, get_emergency_resources
from database import (
    save_assessment, get_user_history, get_trend_data, get_assessment_by_id,
    save_face_assessment, get_face_history,
    save_text_assessment, get_text_history,
    get_all_history, get_all_trend_data
)
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH

# Import emotion and text analysis modules
try:
    from emotion_detector import analyze_emotion, get_emotion_detector
    EMOTION_ENABLED = True
except ImportError as e:
    print(f"⚠ Emotion detection not available: {e}")
    EMOTION_ENABLED = False

try:
    from text_analyzer import analyze_text
    TEXT_ENABLED = True
except ImportError as e:
    print(f"⚠ Text analysis not available: {e}")
    TEXT_ENABLED = False

# -------------------------
# Create Flask App
# -------------------------
app = Flask(__name__)
app.secret_key = "stress_ai_secret_key_123"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Helper function for file uploads
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------
# Load Model & Scaler
# -------------------------
model = joblib.load("models/stress_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# Feature Names (MUST MATCH DATASET ORDER)
# -------------------------
FEATURES = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying"
]

# Feature categories for organized display
FEATURE_CATEGORIES = {
    "psychological": {
        "name": "Psychological Health",
        "icon": "🧠",
        "features": ["anxiety_level", "self_esteem", "mental_health_history", "depression"]
    },
    "physical": {
        "name": "Physical Health",
        "icon": "💪",
        "features": ["headache", "blood_pressure", "sleep_quality", "breathing_problem"]
    },
    "environmental": {
        "name": "Environment",
        "icon": "🏠",
        "features": ["noise_level", "living_conditions", "safety", "basic_needs"]
    },
    "academic": {
        "name": "Academic",
        "icon": "📚",
        "features": ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"]
    },
    "social": {
        "name": "Social",
        "icon": "👥",
        "features": ["social_support", "peer_pressure", "extracurricular_activities", "bullying"]
    }
}

# Feature explanations for the chatbot
FEATURE_EXPLANATIONS = {
    "anxiety_level": "How often do you feel worried, nervous, or on edge? Rate from rarely (1) to very frequently (10).",
    "self_esteem": "How confident do you feel about yourself and your abilities? Low confidence (1) to very confident (10).",
    "mental_health_history": "Have you had previous mental health concerns or treatments? No history (1) to significant history (10).",
    "depression": "How often do you feel sad, hopeless, or lose interest in activities? Rarely (1) to very often (10).",
    "headache": "How frequently do you experience headaches or migraines? Rarely (1) to daily (10).",
    "blood_pressure": "Do you have concerns about high blood pressure? Normal (1) to high concerns (10).",
    "sleep_quality": "How well do you sleep? Poor sleep (1) to excellent, restful sleep (10).",
    "breathing_problem": "Do you experience shortness of breath or breathing difficulties? Never (1) to frequently (10).",
    "noise_level": "How noisy is your living/study environment? Very quiet (1) to very noisy (10).",
    "living_conditions": "How would you rate your living conditions? Poor (1) to excellent (10).",
    "safety": "How safe do you feel in your environment? Unsafe (1) to very safe (10).",
    "basic_needs": "Are your basic needs (food, shelter, clothing) being met? Not met (1) to fully met (10).",
    "academic_performance": "How are you performing academically? Poor (1) to excellent (10).",
    "study_load": "How heavy is your academic workload? Light (1) to overwhelming (10).",
    "teacher_student_relationship": "How is your relationship with teachers/professors? Poor (1) to excellent (10).",
    "future_career_concerns": "How worried are you about your future career? Not worried (1) to very worried (10).",
    "social_support": "How much support do you have from friends and family? None (1) to strong support (10).",
    "peer_pressure": "How much pressure do you feel from peers? None (1) to extreme (10).",
    "extracurricular_activities": "How overwhelmed are you by extracurricular commitments? Not at all (1) to very overwhelmed (10).",
    "bullying": "Have you experienced bullying or harassment? Never (1) to frequently (10)."
}

# -------------------------
# Feature ranges from dataset (min, max)
# User inputs (1-10) will be scaled to these ranges
# -------------------------
FEATURE_RANGES = {
    "anxiety_level": (0, 21),
    "self_esteem": (0, 30),
    "mental_health_history": (0, 1),
    "depression": (0, 27),
    "headache": (0, 5),
    "blood_pressure": (1, 3),
    "sleep_quality": (0, 5),
    "breathing_problem": (0, 5),
    "noise_level": (0, 5),
    "living_conditions": (0, 5),
    "safety": (0, 5),
    "basic_needs": (0, 5),
    "academic_performance": (0, 5),
    "study_load": (0, 5),
    "teacher_student_relationship": (0, 5),
    "future_career_concerns": (0, 5),
    "social_support": (0, 3),
    "peer_pressure": (0, 5),
    "extracurricular_activities": (0, 5),
    "bullying": (0, 5)
}

# Features where higher values mean LESS stress (negative correlation)
# For these, we invert user input: user high (10) -> dataset low value
INVERSE_FEATURES = {
    "self_esteem",
    "sleep_quality", 
    "academic_performance",
    "safety",
    "basic_needs",
    "teacher_student_relationship",
    "social_support",
    "living_conditions"
}

def scale_user_input(feature, value):
    """Scale user input (1-10) to dataset feature range
    For inverse features, higher user input means lower stress, so we invert
    """
    min_val, max_val = FEATURE_RANGES[feature]
    
    if feature in INVERSE_FEATURES:
        # Invert: user 1 -> max, user 10 -> min (high user value = good = low stress)
        scaled = max_val - (value - 1) * (max_val - min_val) / 9
    else:
        # Normal: user 1 -> min, user 10 -> max (high user value = bad = high stress)
        scaled = min_val + (value - 1) * (max_val - min_val) / 9
    
    return scaled

# -------------------------
# ROUTES
# -------------------------

# Landing Page (Feature Showcase)
@app.route("/home")
def home():
    return render_template("index.html")


# Welcome Page (Ask Name)
@app.route("/", methods=["GET", "POST"])
def welcome():
    # Generate session ID if not exists
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    if request.method == "POST":
        session["username"] = request.form["username"]
        return render_template(
            "guide.html",
            username=session["username"],
            step=1
        )
    return render_template("welcome.html")


# Guide Page
@app.route("/guide")
def guide():
    return render_template(
        "guide.html",
        username=session.get("username"),
        step=1
    )


# Measurement Page
@app.route("/measure")
def measure():
    return render_template(
        "measure.html",
        features=FEATURES,
        categories=FEATURE_CATEGORIES,
        explanations=FEATURE_EXPLANATIONS,
        username=session.get("username"),
        step=2
    )


# Result Page
@app.route("/result", methods=["POST"])
def result():
    # Collect user inputs in correct feature order and scale them
    user_inputs = {}
    scaled_inputs = []
    
    for feature in FEATURES:
        raw_value = float(request.form.get(feature))
        user_inputs[feature] = raw_value
        scaled_value = scale_user_input(feature, raw_value)
        scaled_inputs.append(scaled_value)

    # Convert to numpy array
    input_array = np.array(scaled_inputs).reshape(1, -1)

    # Scale input using the trained scaler
    input_scaled = scaler.transform(input_array)

    # Predict stress class (0, 1, 2)
    prediction = model.predict(input_scaled)[0]

    # Map prediction to stress level (dataset uses 0=Low, 1=Moderate, 2=High)
    if prediction == 0:
        stress_level = 1
        stress_text = "Low Stress"
        badge_class = "badge-low"
        progress_class = "low"
    elif prediction == 1:
        stress_level = 2
        stress_text = "Moderate Stress"
        badge_class = "badge-moderate"
        progress_class = "moderate"
    else:
        stress_level = 3
        stress_text = "High Stress"
        badge_class = "badge-high"
        progress_class = "high"

    # Explainable AI – Top contributing features
    importances = model.feature_importances_
    feature_importance = list(zip(FEATURES, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_factors = feature_importance[:3]

    # Generate personalized advice with factor-specific recommendations
    advice = generate_advice(
        stress_level, 
        user_inputs=user_inputs, 
        top_factors=top_factors,
        inverse_features=INVERSE_FEATURES
    )

    # Get emergency resources for high stress
    emergency_resources = get_emergency_resources() if stress_level == 3 else None

    # Save to database
    save_assessment(
        username=session.get("username", "Anonymous"),
        session_id=session.get("session_id", "unknown"),
        stress_level=stress_level,
        stress_text=stress_text,
        inputs=user_inputs,
        top_factors=[(f, float(i)) for f, i in top_factors]
    )

    return render_template(
        "result.html",
        stress_text=stress_text,
        stress_level=stress_level,
        badge_class=badge_class,
        progress_class=progress_class,
        advice=advice,
        top_factors=top_factors,
        emergency_resources=emergency_resources,
        username=session.get("username"),
        step=3
    )


# API endpoint for chatbot explanations
@app.route("/api/explain/<feature>")
def explain_feature(feature):
    explanation = FEATURE_EXPLANATIONS.get(feature, "This factor contributes to your stress assessment.")
    return jsonify({"feature": feature, "explanation": explanation})


# -------------------------
# FACE ANALYSIS ROUTES
# -------------------------

@app.route("/face-upload", methods=["GET", "POST"])
def face_upload():
    """Face image upload page"""
    if not EMOTION_ENABLED:
        return render_template("error.html", 
                               message="Emotion detection is not available. Please install required dependencies.",
                               username=session.get("username"))
    
    if request.method == "POST":
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template("face_upload.html", 
                                   error="No file selected",
                                   username=session.get("username"))
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template("face_upload.html", 
                                   error="No file selected",
                                   username=session.get("username"))
        
        if file and allowed_file(file.filename):
            # Read file bytes for analysis
            image_bytes = file.read()
            
            # Analyze emotion
            result = analyze_emotion(image_bytes=image_bytes)
            
            if not result['success']:
                return render_template("face_upload.html", 
                                       error=result['error'],
                                       username=session.get("username"))
            
            # Save to database
            save_face_assessment(
                username=session.get("username", "Anonymous"),
                session_id=session.get("session_id", "unknown"),
                emotion=result['emotion'],
                confidence=result['confidence'],
                stress_level=result['stress_level'],
                stress_text=result['stress_text'],
                stress_score=result['stress_score'],
                probabilities=result.get('probabilities')
            )
            
            # Generate advice based on emotion
            emotion_advice = get_emotion_advice(result['emotion'], result['stress_level'])
            
            return render_template("face_result.html",
                                   result=result,
                                   advice=emotion_advice,
                                   username=session.get("username"))
        else:
            return render_template("face_upload.html", 
                                   error="Invalid file type. Please upload JPG, PNG, or GIF.",
                                   username=session.get("username"))
    
    return render_template("face_upload.html", username=session.get("username"))


@app.route("/live-face")
def live_face():
    """Live camera face detection page"""
    if not EMOTION_ENABLED:
        return render_template("error.html", 
                               message="Emotion detection is not available.",
                               username=session.get("username"))
    
    return render_template("live_face.html", username=session.get("username"))


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    """API endpoint for live frame analysis"""
    if not EMOTION_ENABLED:
        return jsonify({"success": False, "error": "Emotion detection not available"})
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"})
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        
        # Analyze
        result = analyze_emotion(image_bytes=image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/save-face-result", methods=["POST"])
def save_face_result():
    """Save live face detection result"""
    try:
        data = request.get_json()
        
        save_face_assessment(
            username=session.get("username", "Anonymous"),
            session_id=session.get("session_id", "unknown"),
            emotion=data.get('emotion', 'unknown'),
            confidence=data.get('confidence', 0),
            stress_level=data.get('stress_level', 2),
            stress_text=data.get('stress_text', 'Unknown'),
            stress_score=data.get('stress_score', 5),
            probabilities=data.get('probabilities')
        )
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def get_emotion_advice(emotion, stress_level):
    """Get advice based on detected emotion"""
    advice_map = {
        'angry': [
            "Take a few deep breaths to help calm your nervous system",
            "Try stepping away from the situation for a short break",
            "Consider physical activity to release tension",
            "Write down what's bothering you to process your feelings"
        ],
        'sad': [
            "It's okay to feel sad - allow yourself to experience this emotion",
            "Reach out to a friend or family member for support",
            "Try doing something small that usually brings you joy",
            "Consider journaling about your feelings"
        ],
        'fear': [
            "Ground yourself by focusing on your breath",
            "Identify what's causing your fear - naming it can reduce its power",
            "Talk to someone you trust about your concerns",
            "Remember that fear is temporary and will pass"
        ],
        'happy': [
            "Great to see you're feeling positive!",
            "Consider sharing this good mood with others",
            "Take a moment to appreciate what's going well",
            "Use this energy for something productive you've been putting off"
        ],
        'neutral': [
            "A calm state is great for productivity and focus",
            "Consider using this balanced mood for important decisions",
            "This is a good time for reflection or planning",
            "Maintain this equilibrium with regular self-care"
        ],
        'surprise': [
            "Take a moment to process what surprised you",
            "Surprise can be energizing - channel it positively",
            "Share your experience with someone if it helps",
            "Use this alert state for creative thinking"
        ],
        'disgust': [
            "Identify what's causing this reaction",
            "Remove yourself from the unpleasant situation if possible",
            "Practice self-care to restore your equilibrium",
            "Consider discussing your feelings with someone"
        ]
    }
    
    return advice_map.get(emotion, [
        "Take care of your mental health",
        "Consider practicing mindfulness",
        "Stay connected with supportive people"
    ])


# -------------------------
# TEXT ANALYSIS ROUTES
# -------------------------

@app.route("/text-analysis", methods=["GET", "POST"])
def text_analysis():
    """Text stress analysis page"""
    if not TEXT_ENABLED:
        return render_template("error.html", 
                               message="Text analysis is not available. Please install required dependencies.",
                               username=session.get("username"))
    
    if request.method == "POST":
        text_input = request.form.get("text_input", "").strip()
        
        if not text_input:
            return render_template("text_input.html", 
                                   error="Please enter some text to analyze",
                                   username=session.get("username"))
        
        # Analyze text
        result = analyze_text(text_input)
        
        if not result['success']:
            return render_template("text_input.html", 
                                   error=result['error'],
                                   username=session.get("username"))
        
        # Save to database
        save_text_assessment(
            username=session.get("username", "Anonymous"),
            session_id=session.get("session_id", "unknown"),
            text_input=text_input,
            sentiment_polarity=result['sentiment']['polarity'],
            sentiment_subjectivity=result['sentiment']['subjectivity'],
            stress_level=result['stress_level'],
            stress_text=result['stress_text'],
            stress_score=result['stress_score'],
            keywords_found=result.get('keywords_found'),
            insights=result.get('insights')
        )
        
        return render_template("text_result.html",
                               result=result,
                               text_input=text_input,
                               username=session.get("username"))
    
    return render_template("text_input.html", username=session.get("username"))


# -------------------------
# DASHBOARD ROUTE
# -------------------------

@app.route("/dashboard")
def dashboard():
    """Unified dashboard showing all assessment options"""
    session_id = session.get("session_id", "unknown")
    
    # Get recent history from all types
    all_history = get_all_history(session_id, limit=5)
    trend_data = get_all_trend_data(session_id, limit=20)
    
    # Count assessments by type
    stats = {
        'survey': len([h for h in all_history if h.get('type') == 'survey']),
        'face': len([h for h in all_history if h.get('type') == 'face']),
        'text': len([h for h in all_history if h.get('type') == 'text']),
        'total': len(all_history)
    }
    
    return render_template("dashboard.html",
                           username=session.get("username"),
                           history=all_history,
                           trend_data=trend_data,
                           stats=stats,
                           emotion_enabled=EMOTION_ENABLED,
                           text_enabled=TEXT_ENABLED)


# -------------------------
# UPDATED HISTORY ROUTE
# -------------------------

@app.route("/history")
def history():
    session_id = session.get("session_id", "unknown")
    
    # Get filter parameter
    filter_type = request.args.get('type', 'all')
    
    if filter_type == 'survey':
        history_data = get_user_history(session_id)
        for item in history_data:
            item['type'] = 'survey'
    elif filter_type == 'face':
        history_data = get_face_history(session_id)
    elif filter_type == 'text':
        history_data = get_text_history(session_id)
    else:
        history_data = get_all_history(session_id)
    
    trend_data = get_all_trend_data(session_id)
    
    return render_template(
        "history.html",
        history=history_data,
        trend_data=trend_data,
        filter_type=filter_type,
        username=session.get("username")
    )


# -------------------------
# ERROR HANDLER
# -------------------------

@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", 
                           message="File is too large. Maximum size is 16MB.",
                           username=session.get("username")), 413


# -------------------------
# Run Application
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
