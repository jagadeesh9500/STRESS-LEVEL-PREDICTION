#!/usr/bin/env python
"""Quick test of emotion detector with sample images"""
from emotion_detector import analyze_emotion
import os

test_images = [
    'dataset/train/happy/Training_1206.jpg',
    'dataset/train/angry/Training_3908.jpg', 
    'dataset/train/sad/Training_2913.jpg'
]

for img_path in test_images:
    if os.path.exists(img_path):
        result = analyze_emotion(image_path=img_path)
        print(f"{img_path}: {result.get('emotion', 'N/A')} ({result.get('confidence', 0):.1f}%)")
    else:
        print(f"{img_path}: not found")
