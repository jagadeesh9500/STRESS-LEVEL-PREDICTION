"""
Model Comparison Script
Compares accuracy of all emotion detection models in the models/ folder:
1. emotiondetector.h5 (Keras)
2. facialemotionmodel (1).h5 (Keras)  
3. emotion_model.pt (PyTorch)

Evaluates on the test dataset and reports accuracy metrics.
"""
import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
TEST_DIR = 'dataset/test'
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_LABELS_CAPS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_test_data(max_per_class=100):
    """Load test images from dataset/test folder"""
    images = []
    labels = []
    
    for idx, emotion in enumerate(EMOTION_LABELS):
        emotion_dir = os.path.join(TEST_DIR, emotion)
        if not os.path.exists(emotion_dir):
            print(f"⚠️ Directory not found: {emotion_dir}")
            continue
        
        files = list(Path(emotion_dir).glob('*.jpg')) + list(Path(emotion_dir).glob('*.png'))
        if max_per_class:
            files = files[:max_per_class]
        
        for img_path in files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(idx)
    
    print(f"📊 Loaded {len(images)} test images ({max_per_class} per class)")
    return images, labels


def evaluate_keras_model(model_path, images, labels, input_size=(48, 48)):
    """Evaluate a Keras .h5 model using batch prediction"""
    from tensorflow import keras
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Layers: {len(model.layers)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Batch preprocess all images
    batch_images = []
    for img in images:
        img_resized = cv2.resize(img, input_size)
        img_normalized = img_resized.astype('float32') / 255.0
        batch_images.append(img_normalized)
    
    # Stack into batch array with channel dimension
    batch_array = np.array(batch_images)
    batch_array = np.expand_dims(batch_array, axis=-1)  # Add channel dim
    
    start_time = time.time()
    
    # Batch prediction
    predictions = model.predict(batch_array, verbose=0, batch_size=32)
    predicted_indices = np.argmax(predictions, axis=1)
    
    elapsed = time.time() - start_time
    
    # Calculate accuracy
    labels_array = np.array(labels)
    correct = np.sum(predicted_indices == labels_array)
    total = len(labels)
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(predicted_indices, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"   Inference Time: {elapsed:.2f}s ({elapsed/total*1000:.1f}ms per image)")
    
    print(f"\n   Per-class accuracy:")
    for idx, emotion in enumerate(EMOTION_LABELS_CAPS):
        if class_total[idx] > 0:
            cls_acc = (class_correct[idx] / class_total[idx]) * 100
            print(f"   - {emotion:10s}: {cls_acc:5.1f}% ({class_correct[idx]}/{class_total[idx]})")
    
    return {
        'model': model_path,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'time': elapsed,
        'class_accuracy': {EMOTION_LABELS_CAPS[i]: (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0 
                          for i in range(len(EMOTION_LABELS_CAPS))}
    }


def evaluate_pytorch_model(model_path, images, labels, input_size=(48, 48)):
    """Evaluate a PyTorch .pt model"""
    import torch
    import torch.nn as nn
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Define the model architecture (same as training)
    class EmotionCNN(nn.Module):
        def __init__(self, num_classes=7):
            super(EmotionCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 3 * 3, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EmotionCNN(num_classes=7)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    start_time = time.time()
    
    with torch.no_grad():
        for img, label in zip(images, labels):
            # Preprocess
            img_resized = cv2.resize(img, input_size)
            img_normalized = img_resized.astype('float32') / 255.0
            img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.item()
            
            # Track accuracy
            class_total[label] += 1
            total += 1
            if predicted_idx == label:
                correct += 1
                class_correct[label] += 1
    
    elapsed = time.time() - start_time
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"   Inference Time: {elapsed:.2f}s ({elapsed/total*1000:.1f}ms per image)")
    
    print(f"\n   Per-class accuracy:")
    for idx, emotion in enumerate(EMOTION_LABELS_CAPS):
        if class_total[idx] > 0:
            cls_acc = (class_correct[idx] / class_total[idx]) * 100
            print(f"   - {emotion:10s}: {cls_acc:5.1f}% ({class_correct[idx]}/{class_total[idx]})")
    
    return {
        'model': model_path,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'time': elapsed,
        'class_accuracy': {EMOTION_LABELS_CAPS[i]: (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0 
                          for i in range(len(EMOTION_LABELS_CAPS))}
    }


def main():
    print("🔬 Emotion Detection Model Comparison")
    print("="*60)
    
    # Load test data
    images, labels = load_test_data(max_per_class=None)  # Use all test images
    
    if len(images) == 0:
        print("❌ No test images found!")
        return
    
    results = []
    
    # Evaluate Keras models
    keras_models = [
        'models/emotiondetector.h5',
        'models/facialemotionmodel (1).h5'
    ]
    
    for model_path in keras_models:
        result = evaluate_keras_model(model_path, images, labels)
        if result:
            results.append(result)
    
    # Evaluate PyTorch model
    pytorch_result = evaluate_pytorch_model('models/emotion_model.pt', images, labels)
    if pytorch_result:
        results.append(pytorch_result)
    
    # Summary
    print("\n" + "="*60)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*60)
    
    if results:
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("\n🏆 Ranking by Accuracy:\n")
        for i, r in enumerate(results, 1):
            model_name = os.path.basename(r['model'])
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} #{i}: {model_name}")
            print(f"       Accuracy: {r['accuracy']:.2f}%")
            print(f"       Speed: {r['time']/r['total']*1000:.1f}ms/image")
            print()
        
        best_model = results[0]
        print(f"\n✅ RECOMMENDED MODEL: {os.path.basename(best_model['model'])}")
        print(f"   Accuracy: {best_model['accuracy']:.2f}%")
        
        return best_model
    else:
        print("❌ No models could be evaluated!")
        return None


if __name__ == '__main__':
    main()
