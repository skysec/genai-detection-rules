"""
Positive test cases for TensorFlow detection rule
These should all be detected by the detect-tensorflow rule
"""

# Test 1: Import tensorflow
# ruleid: detect-tensorflow
import tensorflow

# Test 2: From tensorflow import
# ruleid: detect-tensorflow
from tensorflow import keras

# Test 3: Import tensorflow.keras
# ruleid: detect-tensorflow
import tensorflow.keras

# Test 4: Constant tensor
# ruleid: detect-tensorflow
constant = tensorflow.constant([1, 2, 3])

# Test 5: Variable
# ruleid: detect-tensorflow
var = tensorflow.Variable([1.0, 2.0])

# Test 6: Zeros
# ruleid: detect-tensorflow
zeros = tensorflow.zeros([3, 3])

# Test 7: Sequential model
# ruleid: detect-tensorflow
model = tensorflow.keras.Sequential()

# Test 8: Dense layer
# ruleid: detect-tensorflow
dense = tensorflow.keras.layers.Dense(10)

# Test 9: Conv2D layer
# ruleid: detect-tensorflow
conv = tensorflow.keras.layers.Conv2D(32, (3, 3))

# Test 10: LSTM layer
# ruleid: detect-tensorflow
lstm = tensorflow.keras.layers.LSTM(64)

# Test 11: Adam optimizer
# ruleid: detect-tensorflow
optimizer = tensorflow.keras.optimizers.Adam()

# Test 12: CrossEntropy loss
# ruleid: detect-tensorflow
loss = tensorflow.keras.losses.SparseCategoricalCrossentropy()

# Test 13: Model compile
# ruleid: detect-tensorflow
model.compile(optimizer='adam', loss='mse')

# Test 14: Model fit
# ruleid: detect-tensorflow
model.fit(x_train, y_train, epochs=10)

# Test 15: Model save
# ruleid: detect-tensorflow
model.save('my_model')

# Test 16: Load model
# ruleid: detect-tensorflow
loaded_model = tensorflow.keras.models.load_model('my_model')

# Test 17: ModelCheckpoint callback
# ruleid: detect-tensorflow
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint('model.h5')

# Test 18: Dataset from_tensor_slices
# ruleid: detect-tensorflow
dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))

# Test 19: GradientTape
# ruleid: detect-tensorflow
with tensorflow.GradientTape() as tape:
    loss = loss_fn(y_true, y_pred)

# Test 20: Real-world model
def create_model():
    # ruleid: detect-tensorflow
    model = tensorflow.keras.Sequential([
        # ruleid: detect-tensorflow
        tensorflow.keras.layers.Dense(128, activation='relu'),
        # ruleid: detect-tensorflow
        tensorflow.keras.layers.Dropout(0.2),
        # ruleid: detect-tensorflow
        tensorflow.keras.layers.Dense(10, activation='softmax')
    ])

    # ruleid: detect-tensorflow
    model.compile(
        # ruleid: detect-tensorflow
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy'
    )

    return model
