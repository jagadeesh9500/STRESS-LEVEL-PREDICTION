import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
import cv2

# Load model
model = keras.models.load_model('models/emotiondetector.h5', compile=False)
model.summary()

# Test on sample images
print("\n\nTesting on sample images:")
test_emotions = ['angry', 'happy', 'sad', 'neutral']
for emotion in test_emotions:
    test_dir = f'dataset/test/{emotion}'
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
        for f in files:
            img = cv2.imread(os.path.join(test_dir, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (48, 48))
                img_normalized = img_resized.astype('float32') / 255.0
                img_input = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)
                predictions = model.predict(img_input, verbose=0)[0]
                pred_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(predictions)]
                print(f"  {emotion}/{f}: Predicted={pred_label}, Probs={predictions.round(3)}")
