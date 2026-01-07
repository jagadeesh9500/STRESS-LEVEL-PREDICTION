#!/usr/bin/env python3
"""Check Emotion_Detection.h5 model using tf.keras"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

model_path = 'models/Emotion_Detection.h5'
print(f"Loading model: {model_path}")
print(f"TensorFlow version: {tf.__version__}")

try:
    # Try loading with custom object scope for older models
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
    
    # Load with legacy support
    model = tf.keras.models.load_model(
        model_path, 
        compile=False,
        safe_mode=False
    )
    print(f"Success! Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Test prediction
    test_input = np.zeros((1, 48, 48, 1), dtype=np.float32)
    pred = model.predict(test_input, verbose=0)
    print(f"Test prediction shape: {pred.shape}")
    print(f"Test prediction: {pred}")
    
except Exception as e:
    print(f"Error: {e}")
    
    print("\nTrying h5py to inspect file structure...")
    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            print("Model file keys:", list(f.keys()))
            if 'model_weights' in f:
                print("Weights keys:", list(f['model_weights'].keys()))
    except Exception as e2:
        print(f"h5py error: {e2}")
