"""
Negative test cases for TensorFlow detection rule
These should NOT be detected by the detect-tensorflow rule
"""

# Test 1: Comments mentioning tensorflow
# This code uses tensorflow but doesn't import it

# Test 2: Strings containing tensorflow
framework = "tensorflow"
description = "This uses TensorFlow"

# Test 3: Dictionary with tensorflow keys
config = {
    "framework": "tensorflow",
    "backend": "keras"
}

# Test 4: Variable names
tensorflow_enabled = True
use_tensorflow = False
keras_model = "model.h5"

# Test 5: URLs
docs_url = "https://www.tensorflow.org/docs"
keras_url = "https://keras.io"

# Test 6: Environment variables
import os
TF_CPP_MIN_LOG_LEVEL = os.getenv("TF_CPP_MIN_LOG_LEVEL")

# Test 7: Mock model class
class Sequential:
    """Mock Sequential - not TensorFlow"""
    def __init__(self, layers=None):
        self.layers = layers or []

    def compile(self, optimizer, loss):
        pass

    def fit(self, x, y, epochs):
        pass

model = Sequential()
model.compile(optimizer='adam', loss='mse')
model.fit([1, 2, 3], [4, 5, 6], epochs=10)
