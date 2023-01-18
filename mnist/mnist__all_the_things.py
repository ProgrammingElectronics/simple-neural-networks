import numpy as np
import matplotlib.pyplot as plot
from keras.datasets import mnist

# Activation Functions
def relu(x):
    return (x >= 0) * x

def relu_deriv(x):
    return (x >= 0)

# Create Embeddings
def create_embedding(val):
    embedding = np.zeros((1,10))
    embedding[0,val] = 1
    return embedding

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Transformation
training_size = 1000

## Training Data
labels = np.array([create_embedding(y) for y in y_train[:training_size]]).reshape((training_size,10))
images = x_train[:training_size].reshape(training_size,784) / 255

assert labels.shape == (training_size, 10) , f"Expect (1000, 10), actual {labels.shape}"
assert images.shape == (training_size, 784) , f"Expect (1000, 784, actual {images.shape}"

## Test Data
test_labels = np.array([create_embedding(y) for y in y_test]).reshape((len(y_test),10))
test_images = x_test.reshape(len(x_test),784) / 255

assert test_labels.shape == (len(y_test), 10), f"Expect (10000, 10), actual {test_labels.shape}"
assert test_images.shape == (len(x_test), 784) , f"Expect (1000, 784, actual {test_images.shape}"

# Controls
alpha = 0.1
epochs = 300
mini_batch_size = 100

# Network Architecture
hidden_size = 40

# Weights
w_0_1 = 2 * np.random.random((784, hidden_size)) - 1
w_1_2 = 2 * np.random.random((hidden_size, 10)) - 1

# Training
for epoch in range(epochs):

    pass
    # Predict

    # Mini-batch

    # Drop Out

    # Compare

    # Error and Accuracy

    # Update Weights

# Testing

    # Predict

    # Compare

# Printing Results

# Graphing Results
