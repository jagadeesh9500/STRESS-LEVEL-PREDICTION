"""
Emotion Detection Module using TensorFlow/Keras
Detects facial emotions from images using pre-trained CNN model (emotiondetector.h5)

Model: 18-layer CNN trained on FER2013 dataset
Input: 48x48 grayscale images
Output: 7 emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
"""
import os
import numpy as np
import cv2
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============== Configuration ==============
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Model path
EMOTION_MODEL_PATH = 'models/facialemotionmodel.h5'

# Emotion to stress score mapping (1-10 scale)
EMOTION_STRESS_MAP = {
    'Angry': 8,
    'Disgust': 6,
    'Fear': 9,
    'Happy': 1,
    'Sad': 7,
    'Surprise': 3,
    'Neutral': 4
}

# Stress category mapping (text, level)
EMOTION_STRESS_CATEGORY = {
    'Angry': ('High Stress', 3),
    'Disgust': ('Moderate Stress', 2),
    'Fear': ('High Stress', 3),
    'Happy': ('Low Stress', 1),
    'Sad': ('High Stress', 3),
    'Surprise': ('Low Stress', 1),
    'Neutral': ('Moderate Stress', 2)
}

# Emotion descriptions
EMOTION_DESCRIPTIONS = {
    'Angry': 'Signs of frustration or anger detected. This emotional state is often associated with elevated stress levels.',
    'Disgust': 'Expression indicates displeasure or aversion. May suggest moderate stress or discomfort.',
    'Fear': 'Anxiety or fearful expression detected. This is typically associated with high stress levels.',
    'Happy': 'Positive, happy expression detected! This indicates low stress and good emotional well-being.',
    'Sad': 'Sadness indicators present. Prolonged sadness can contribute to elevated stress levels.',
    'Surprise': 'Surprised expression detected. This is typically a neutral emotion with mild stress impact.',
    'Neutral': 'Neutral, calm expression detected. Your emotional state appears balanced and stable.'
}

# Emoji mapping
EMOTION_EMOJI = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}


class EmotionDetector:
    """
    Emotion detector using TensorFlow/Keras CNN model
    Uses pre-trained emotiondetector.h5 model (48x48 grayscale input, 7 emotion outputs)
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion detector
        
        Args:
            model_path: Path to .h5 model file (optional, uses default if not provided)
        """
        self.model = None
        self.model_path = model_path or EMOTION_MODEL_PATH
        self.model_name = "FacialEmotionModel CNN (Keras)"
        self.input_size = (48, 48)
        self.face_cascade = None
        self.is_loaded = False
        
        self._load_model()
        self._load_face_detector()
    
    def _load_model(self):
        """Load the Keras model from .h5 file"""
        try:
            from tensorflow import keras
            
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path, compile=False)
                self.is_loaded = True
                print(f"✅ Loaded emotion model: {self.model_path}")
                print(f"   Architecture: 4-Conv CNN with Dropout")
                print(f"   Input: {self.input_size[0]}x{self.input_size[1]} grayscale")
                print(f"   Output: {len(EMOTION_LABELS)} emotions")
            else:
                print(f"⚠️ Model not found: {self.model_path}")
                self.is_loaded = False
                
        except ImportError as e:
            print(f"❌ TensorFlow not installed: {e}")
            print("   Install with: pip install tensorflow")
            self.is_loaded = False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def _load_face_detector(self):
        """Load OpenCV Haar Cascade for face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("⚠️ Could not load face cascade classifier")
        except Exception as e:
            print(f"⚠️ Error loading face detector: {e}")
    
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        
        Args:
            face_img: Grayscale face image (numpy array)
            
        Returns:
            Preprocessed image ready for prediction (batch, height, width, channels)
        """
        # Resize to model input size
        face_resized = cv2.resize(face_img, self.input_size)
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Add batch and channel dimensions: (1, 48, 48, 1)
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)
        
        return face_input
    
    def detect_faces(self, image):
        """
        Detect faces in an image using Haar Cascade
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        if self.face_cascade is None or image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return list(faces) if len(faces) > 0 else []
    
    def predict_emotion(self, face_image):
        """
        Predict emotion from a face image
        
        Args:
            face_image: Grayscale face image (numpy array) or PIL Image
            
        Returns:
            tuple: (emotion_label, confidence, all_probabilities_dict)
        """
        if not self.is_loaded or self.model is None:
            # Return default prediction if model not loaded
            return 'Neutral', 50.0, {label: 14.28 for label in EMOTION_LABELS}
        
        # Convert PIL to numpy if needed
        if hasattr(face_image, 'convert'):
            face_image = np.array(face_image.convert('L'))
        
        # Ensure grayscale
        if len(face_image.shape) == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Preprocess
        face_input = self.preprocess_face(face_image)
        
        # Predict
        predictions = self.model.predict(face_input, verbose=0)[0]
        
        # Get emotion with highest probability
        emotion_idx = np.argmax(predictions)
        confidence = float(predictions[emotion_idx]) * 100
        emotion = EMOTION_LABELS[emotion_idx]
        
        # Create probabilities dict
        all_probs = {label: round(float(predictions[i]) * 100, 2) 
                     for i, label in enumerate(EMOTION_LABELS)}
        
        return emotion, confidence, all_probs
    
    def analyze_image(self, image_path=None, image_bytes=None, image_array=None):
        """
        Analyze an image and return emotion detection results
        
        Args:
            image_path: Path to image file
            image_bytes: Raw image bytes (for file uploads)
            image_array: BGR numpy array (for webcam frames)
            
        Returns:
            dict with detection results
        """
        # Load image based on input type
        if image_path:
            image = cv2.imread(image_path)
        elif image_bytes:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_array is not None:
            image = image_array
        else:
            return self._error_result("No image provided")
        
        if image is None:
            return self._error_result("Could not read image")
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return self._error_result("No face detected in the image. Please ensure your face is clearly visible.")
        
        # Use the largest face (most likely the main subject)
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        
        x, y, w, h = faces[0]
        
        # Extract face region (grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict emotion
        emotion, confidence, all_probs = self.predict_emotion(face_roi)
        
        # Get stress mapping
        stress_score = EMOTION_STRESS_MAP.get(emotion, 5)
        stress_text, stress_level = EMOTION_STRESS_CATEGORY.get(emotion, ('Moderate Stress', 2))
        description = EMOTION_DESCRIPTIONS.get(emotion, '')
        emoji = EMOTION_EMOJI.get(emotion, '😐')
        
        return {
            'success': True,
            'emotion': emotion,
            'emoji': emoji,
            'confidence': round(confidence, 1),
            'probabilities': all_probs,
            'stress_score': stress_score,
            'stress_level': stress_level,
            'stress_text': stress_text,
            'description': description,
            'model_name': self.model_name,
            'face_count': len(faces),
            'face_location': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        }
    
    def analyze_frame(self, frame):
        """
        Analyze a video frame (for live detection)
        Same as analyze_image but optimized for real-time
        """
        return self.analyze_image(image_array=frame)
    
    def _error_result(self, message):
        """Create standardized error response"""
        return {
            'success': False,
            'error': message,
            'emotion': None,
            'confidence': 0,
            'stress_score': 5,
            'stress_level': 2,
            'stress_text': 'Moderate Stress',
            'model_name': self.model_name
        }


# Singleton instance for reuse
_detector_instance = None


def get_emotion_detector():
    """Get singleton EmotionDetector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmotionDetector()
    return _detector_instance


def analyze_emotion(image_path=None, image_bytes=None):
    """
    Convenience function to analyze emotion from an image
    
    Args:
        image_path: Path to image file
        image_bytes: Raw image bytes
        
    Returns:
        dict with emotion detection results
    """
    detector = get_emotion_detector()
    return detector.analyze_image(image_path=image_path, image_bytes=image_bytes)


def get_model_info():
    """Get information about the loaded model"""
    info = {
        'name': 'facialemotionmodel.h5',
        'type': 'Keras/TensorFlow CNN',
        'architecture': '4-Conv CNN with Dropout (4.2M params)',
        'input_size': '48x48 grayscale',
        'output': f'{len(EMOTION_LABELS)} emotions',
        'emotions': EMOTION_LABELS,
        'status': 'not_found'
    }
    
    if os.path.exists(EMOTION_MODEL_PATH):
        size_mb = os.path.getsize(EMOTION_MODEL_PATH) / (1024 * 1024)
        info['size_mb'] = round(size_mb, 2)
        info['status'] = 'loaded'
    
    return info


# Export public API
__all__ = [
    'EmotionDetector',
    'get_emotion_detector', 
    'analyze_emotion',
    'get_model_info',
    'EMOTION_LABELS',
    'EMOTION_STRESS_MAP',
    'EMOTION_STRESS_CATEGORY',
    'EMOTION_EMOJI',
    'EMOTION_DESCRIPTIONS'
]
