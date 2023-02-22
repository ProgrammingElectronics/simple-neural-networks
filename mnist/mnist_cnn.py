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

# convolution 
def get_image_section(layer, start_row, stop_row,start_col,stop_col):
    
    section = layer[start_row:stop_row, start_col:stop_col]
    return section.reshape(-1, 1, start_row-start_row, start_col-stop_col)


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
epochs = 2
alpha = 2
mini_batch_size = 100

## Architecture 
input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16
num_labels = 10
hidden_layer_size = (input_rows - kernel_rows) * (input_cols - kernel_cols)

## Weights ###################
kernels = 0.02*np.random.random((kernel_rows*kernel_cols, num_kernels))-0.01
wt_1_2 = 0.2*np.random.random((hidden_layer_size, num_labels))-0.1

## Visualization 
plot_freq = 10
accuracy_plot = np.zeros(int(epochs/plot_freq))

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
