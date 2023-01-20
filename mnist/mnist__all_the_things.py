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
epochs = 1
mini_batch_size = 100

# Network Architecture
hidden_size = 40

# Weights
w_0_1 = 2 * np.random.random((784, hidden_size)) - 1
w_1_2 = 2 * np.random.random((hidden_size, 10)) - 1

# Training
for epoch in range(epochs):

    # Metrics
    error = 0
    correct_count = 0

    # Mini-batch
    for i in range(int(len(labels)/mini_batch_size)):
        
        # Mini Batch
        images_mini_batch = images[i * mini_batch_size:i * mini_batch_size + mini_batch_size]
        labels_mini_batch = labels[i * mini_batch_size:i * mini_batch_size + mini_batch_size]

        # Drop Out Mask
        drop_out_mask = np.random.randint(2, size=(mini_batch_size,hidden_size))

        # Predict #
        l0 = images_mini_batch
        l1 = relu(l0.dot(w_0_1)) * drop_out_mask
        l2 = l1.dot(w_1_2)
        
        assert l0.shape == (mini_batch_size,784) , f"Expect ({mini_batch_size}, 784), actual {l0.shape}" 
        assert l1.shape == (100,hidden_size) , f"Expect (100,{hidden_size}), actual {l1.shape}"
        assert l2.shape == (100,10) , f"Expect (100,10), actual {l2.shape}"

        # Compare #
        # Error and Accuracy
        error += np.sum((l2 - labels_mini_batch) ** 2)
        
        for i in range(mini_batch_size):
            correct_count += np.argmax(labels_mini_batch[i]) == np.argmax(l2[i])
            
        # Update Weights
        l2_delta = (l2 - labels_mini_batch) # prediction - actual
        l1_delta = l2_delta.dot(w_1_2.T)*relu_deriv(l1) 
        
        l2_wt_delta = l2_delta.T.dot(l1)
        l1_wt_delta = l1_delta.T.dot(l0)

        w_1_2 -= l2_wt_delta.T * alpha
        w_0_1 -= l1_wt_delta.T * alpha
# Testing

    # Predict

    # Compare

# Printing Results

# Graphing Results
