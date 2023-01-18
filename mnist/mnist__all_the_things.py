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
labels = np.array([create_embedding(y) for y in y_train[:training_size]]).reshape((training_size,10))
assert labels.shape == (training_size, 10) , f"Expect (1000, 10), actual {labels.shape}"


# Controls


# Network Architecture

# Weights

# Training

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
