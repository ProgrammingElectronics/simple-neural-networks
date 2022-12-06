from keras.datasets import mnist
import numpy as np

def create_embedding(val):
    embedding = np.zeros((10))
    embedding[val] = 1
    result = np.reshape(embedding, (1,10))
    return result

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Get single data point
images = x_train [0:1]
labels = y_train[0:1]

# Get just first image
image = images[0]

# Concatenate image into a 1 by 784 len array
flat_image = np.reshape(image, (1,784)) # (784,1)

# Data
input = flat_image
actual = create_embedding(labels) # (1,10)

# Weights
weights = np.random.random((784,10))

# Knobs
epochs = 20
alpha = 0.0000001

for epoch in range(epochs):

    # Predict
    pred = input.dot(weights) # (1, 784) dot (784,10) -> (1,10)
    
    # Compare
    mse = (pred - actual) ** 2 # -> (1,10)
    delta = pred - actual # -> (1, 10)
    
    # Adjust Weights
    weight_delta = delta.T * input # (1,10) * (1,784) -> (10, 784) 
    weights -= weight_delta.T * alpha # (784,10) - (10,784) -> (784,10)
    
    print("Error" + str(mse))
    print("Pred" + str(pred))