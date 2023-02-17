## imports #################
import numpy as np
import matplotlib.pyplot as plot
from keras.datasets import mnist

## Functions #################

# one hot embedding
def embedding(x):

    embedding = np.zeros((1,10))
    embedding[0,x] = 1
    return embedding

# tanh
def tanh(x):
    
    return np.tanh(x)

# tanh derivative
def tanh_deriv(x):

    return 1 - (x ** 2)

# softmax
def softmax(x):

    temp = np.exp(x)
    return temp / np.sum(temp)

## Data #######################
# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform data
training_size = 1000
# train data
images = x_train[:training_size].reshape((training_size,784)) / 255
labels = np.array([embedding(y) for y in y_train[:training_size]]).reshape((training_size,-1))

# test data
test_images = x_test.reshape((len(x_test),784)) / 255
test_labels = np.array([embedding(y) for y in y_test]).reshape((len(y_test), -1))

## Controls ###################

## Architecture 

## Weights ###################

## Visualization 

## Training ###################

    # Mini Batch 
    # Drop Out
    
    ## Predict
    # Forward Prop with convolution

    ## Compare
    # NO ERROR
    # Accuracy

    ## Adjust
    # Delta back prop
    # Adjust Weights

## Testing ###################

    ## Predict
    # Forward Prop with convolution
    
    ## Compare
    # Accuracy


## Visualize ###################

# console
# accuracy plots
# weight color plots?
