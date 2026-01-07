Stress Level Prediction & Mental Health Advisor

A comprehensive web-based application that predicts stress levels using multiple analysis methods and provides personalized mental health advice. This system combines survey-based assessment, facial emotion detection, and text sentiment analysis to evaluate stress levels and offer tailored recommendations.
🎯 Project Overview

This application combines machine learning with mental health guidance to:

    Multi-Modal Stress Assessment: Three ways to measure stress (survey, face, text)
    Facial Emotion Detection: AI-powered analysis of facial expressions using CNN
    Text Sentiment Analysis: NLP-based stress detection from written text
    Survey-Based Prediction: Analyze 20 stress-related factors
    Provide Personalized Advice: Generate actionable recommendations based on stress level
    Identify Key Factors: Use explainable AI to highlight the most significant stress contributors
    Track History: SQLite database stores assessment history for progress tracking
    User-Friendly Interface: Interactive web interface with intuitive input forms

🏗️ Project Structure

├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── database.py                     # SQLite database operations
├── emotion_detector.py             # Facial emotion detection module
├── text_analyzer.py                # Text sentiment analysis module
├── advice_engine.py                # Advice generation logic
├── model_training.py               # ML model training script
├── stress_prediction.py            # Stress prediction testing module
├── train_emotion_model.py          # Emotion model training script
├── check_setup.py                  # Environment validation
├── test_dataset.py                 # Dataset verification
├── stress_history.db               # SQLite database for history
├── models/
│   ├── stress_model.pkl            # Trained RandomForest model
│   ├── scaler.pkl                  # StandardScaler for feature normalization
│   ├── emotion_model.pt            # PyTorch emotion model
│   ├── emotiondetector.h5          # Keras emotion detector model
│   └── facialemotionmodel.h5       # Keras facial emotion CNN model
├── templates/
│   ├── welcome.html                # User welcome & name entry
│   ├── guide.html                  # Information guide
│   ├── measure.html                # Stress measurement form (survey)
│   ├── result.html                 # Survey prediction results & advice
│   ├── index.html                  # Landing page / feature showcase
│   ├── dashboard.html              # Unified dashboard
│   ├── history.html                # Assessment history view
│   ├── face_upload.html            # Face image upload page
│   ├── face_result.html            # Face analysis results
│   ├── live_face.html              # Live camera emotion detection
│   ├── text_input.html             # Text input for sentiment analysis
│   ├── text_result.html            # Text analysis results
│   └── error.html                  # Error display page
├── static/
│   ├── style.css                   # Custom styling
│   └── images/                     # UI images
├── uploads/                        # Uploaded face images
└── README.md                       # This file

📊 Features
🎭 Three Assessment Methods
1. Survey-Based Assessment (20 Factors)

Rate yourself on 20 stress-related factors across multiple dimensions:

Psychological Factors

    Anxiety Level
    Self-Esteem
    Mental Health History
    Depression

Physical Health Indicators

    Headache
    Blood Pressure
    Sleep Quality
    Breathing Problems

Environmental Factors

    Noise Level
    Living Conditions
    Safety
    Basic Needs

Academic/Professional Factors

    Academic Performance
    Study Load
    Teacher-Student Relationship
    Future Career Concerns

Social Factors

    Social Support
    Peer Pressure
    Extracurricular Activities
    Bullying

2. Facial Emotion Detection

    Upload Image: Upload a photo for emotion analysis
    Live Camera: Real-time emotion detection using webcam
    CNN Model: Uses 18-layer CNN trained on FER2013 dataset
    7 Emotions: Detects Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    Emotion-to-Stress Mapping: Maps detected emotion to stress level

3. Text Sentiment Analysis

    NLP-Powered: Uses TextBlob for sentiment analysis
    Keyword Detection: Identifies stress-related keywords
    Polarity Analysis: Measures positive/negative sentiment
    Contextual Understanding: Considers text subjectivity

Stress Levels

The system classifies stress into three categories:

    Low Stress (Level 1): You are doing well
    Moderate Stress (Level 2): Maintain work-life balance
    High Stress (Level 3): Professional help recommended

📈 Dashboard & History

    Unified Dashboard: View all assessment types in one place
    History Tracking: SQLite database stores all assessments
    Trend Analysis: Track stress levels over time
    Filter by Type: View survey, face, or text assessments separately

Personalized Recommendations

Each stress level includes tailored advice:

    Low Stress: Maintain healthy habits
    Moderate Stress: Engagement in physical activities, regular breaks
    High Stress: Sleep improvement, meditation, workload reduction, professional consultation

Explainable AI

The results page displays the top 3 contributing factors to the stress prediction, helping users understand which aspects most influenced their score.
🛠️ Technology Stack

    Backend: Flask (Python web framework)
    Machine Learning:
        scikit-learn (Random Forest Classifier)
        TensorFlow/Keras (CNN for emotion detection)
        PyTorch (alternative emotion model)
    Computer Vision: OpenCV, Pillow
    NLP: TextBlob for sentiment analysis
    Database: SQLite for history tracking
    Frontend: Bootstrap 5, HTML5, CSS3, JavaScript
    Data Processing: Pandas, NumPy
    Model Persistence: joblib, .h5, .pt formats

🚀 Getting Started
Prerequisites

    Python 3.7+
    pip (Python package manager)
    Webcam (optional, for live face detection)

Required Libraries

flask
werkzeug
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn
torch
torchvision
tensorflow
keras
opencv-python-headless
Pillow
textblob

Installation

Clone or download the project

git clone https://github.com/jagadeesh9500/Stress-level-Prediction.git
cd Stress-Level-Prediction-

Create virtual environment (recommended)

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

Install dependencies

pip install -r requirements.txt

Download TextBlob corpora

python -m textblob.download_corpora

Verify setup (check all dependencies are installed)

python check_setup.py

Train the stress model (if models/ folder doesn't have stress_model.pkl)

python model_training.py

📈 Running the Application

Start the Flask server

python app.py

    Open your browser Navigate to: http://localhost:5000

    Use the Application
        Enter your name on the welcome page
        Choose your assessment method:
            Survey: Rate 20 stress factors (1-10)
            Face Upload: Upload an image for emotion analysis
            Live Camera: Real-time emotion detection
            Text Analysis: Write about your feelings
        View personalized prediction and advice
        Check your history and trends on the dashboard

📝 How It Works
1. Survey-Based Assessment

    Users rate themselves on 20 stress-related factors (1-10 scale)
    Values are scaled to match dataset feature ranges
    Random Forest model classifies stress level
    Feature importance identifies top contributing factors

2. Facial Emotion Detection

    CNN model (FER2013-trained) processes 48x48 grayscale images
    Detects one of 7 emotions with confidence score
    Maps emotion to stress level (e.g., Angry → High Stress)
    Works with uploaded images or live webcam feed

3. Text Sentiment Analysis

    TextBlob analyzes sentiment polarity (-1 to +1)
    Keyword detection identifies stress-related words
    Subjectivity score measures emotional vs factual content
    Combined analysis produces stress prediction

4. Advice Generation

Based on the predicted stress level, the system generates relevant recommendations tailored to the assessment method and detected factors.
5. History & Trends

All assessments are stored in SQLite database for tracking progress over time.
🔧 Key Files Description
File 	Purpose
app.py 	Main Flask application with all route handlers
config.py 	Configuration settings (paths, thresholds, mappings)
database.py 	SQLite database operations for history tracking
emotion_detector.py 	Facial emotion detection using CNN/Keras
text_analyzer.py 	Text sentiment analysis using TextBlob
advice_engine.py 	Generates personalized advice based on stress level
model_training.py 	Trains Random Forest model on stress dataset
train_emotion_model.py 	Trains emotion detection model
check_setup.py 	Validates all required libraries are installed
test_dataset.py 	Verifies dataset integrity and structure
📊 Model Details
Stress Prediction Model

Algorithm: Random Forest Classifier

    Estimators: 300 trees
    Random State: 42 (for reproducibility)
    Class Weights: Balanced (handles class imbalance)
    Train-Test Split: 80-20 ratio
    Scaling: StandardScaler

Emotion Detection Model

Architecture: 18-layer CNN (Keras/TensorFlow)

    Input: 48x48 grayscale images
    Output: 7 emotion classes
    Training Data: FER2013 dataset
    Format: .h5 (Keras model)

Text Analysis

Library: TextBlob

    Polarity Range: -1 (negative) to +1 (positive)
    Subjectivity Range: 0 (objective) to 1 (subjective)

🌐 Web Routes
Route 	Method 	Description
/ 	GET, POST 	Welcome page and username entry
/home 	GET 	Landing page with feature showcase
/guide 	GET 	Information guide about stress factors
/measure 	GET 	Stress measurement form (20 questions)
/result 	POST 	Displays survey predictions and advice
/face-upload 	GET, POST 	Face image upload for emotion analysis
/live-face 	GET 	Live camera emotion detection
/text-analysis 	GET, POST 	Text sentiment analysis
/dashboard 	GET 	Unified dashboard with all assessments
/history 	GET 	Assessment history with filters
/api/analyze-frame 	POST 	API for live frame analysis
/api/save-face-result 	POST 	API to save face detection results
/api/explain/<feature> 	GET 	API for feature explanations
🛡️ Security Features

    Session-based username storage
    Secret key configuration for Flask session management
    Input validation and type conversion
    Secure file upload handling with allowed extensions
    Maximum file size limit (16MB)

💡 Usage Tips

    Honest Assessment: Rate factors based on your actual experience
    Context: Consider the past week or month when rating
    Professional Help: For high stress, consider consulting a mental health professional
    Regular Monitoring: Use periodically to track stress changes

📌 Notes

    The stress model requires a properly formatted CSV dataset with the 20 features and a stress_level column
    Feature order in the application must match the dataset order
    Emotion detection requires the .h5 model files in the models/ directory
    Webcam access is required for live face detection feature
    Text analysis requires TextBlob corpora to be downloaded
    All assessment data is stored locally in SQLite database

🔮 Future Enhancements

    User authentication and accounts
    Cloud database integration
    Multiple language support
    Enhanced visualizations and charts
    Integration with mental health resources
    Mobile application version
    Push notifications for stress alerts
    Voice-based stress analysis
    Wearable device integration
    Export reports as PDF

📄 License

This project is created for educational and mental health assessment purposes.
✍️ Author

jagadeesh9500

Developed as a comprehensive stress prediction and mental health advisory system.

⭐ If you find this project helpful, please consider giving it a star
