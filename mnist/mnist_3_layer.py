#############################################
# L_0    ## (784,1)         -> input
# WT_0_1 ## (784,100)
# L_1    ## (100,1)         -> hidden layer
# WT_1_2 ## (100,10)
# L_2    ## (10,1)          -> output
############################################

import numpy as np
import matplotlib.pyplot as plt

# Import mnist data set
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.random.seed(42)

def createEmbedding(val):

    embedding = np.zeros((1, 10))
    embedding[0, val] = 1
    return embedding

# create activation RELU function for prediction
def relu(x):

    return (x > 0) * x

# create derivative RELU function for back-propobation
def relu_deriv(x):

    return (x > 0)

### Network Layout ###
input_layer_nodes = 784
hidden_layer_nodes = 100
output_layer_nodes = 10

### Adjustments ###
alpha = 0.000000001
epochs = 100

## Prepare data ###
inputs = x_train.reshape((60000, 784))

labels = np.zeros((60000, 10))
for i, label in enumerate(y_train):
    labels[i] = createEmbedding(label)

### Create weight arrays for WT_0_1 and WT_1_2 ###
wt_0_1 = 2*np.random.random((input_layer_nodes,
                            hidden_layer_nodes))-1
wt_1_2 = 2*np.random.random((hidden_layer_nodes,
                            output_layer_nodes))-1  # (100,10)


# For ploting error
y_axis_error = np.zeros((1,epochs))
x_axis_epochs = np.arange(0,epochs).reshape((1,epochs))


for epoch in range(epochs):
    
    error = 0

    for input, label in zip(inputs, labels):

        # Prep input and label
        l0 = input.reshape((1, 784)).astype('float64')
        actual = label.reshape((1, 10))

        ### Predict L1 and L2
        l1 = relu(l0.dot(wt_0_1))
        l2 = l1.dot(wt_1_2)

        #### Compare -> calculate MSE
        error = np.sum((l2 - actual) ** 2).astype('float64')
        
        # Calculate delta for l2 and l1
        l2_delta = l2 - actual
        l1_delta = l2_delta.dot(wt_1_2.T) * relu_deriv(l1)
    
        # Calulate weight delta
        l2_wt_delta = l2_delta.T.dot(l1)
        l1_wt_delta = l1_delta.T.dot(l0)

        # Adjust weights
        wt_1_2 -= l2_wt_delta.T * alpha
        wt_0_1 -= l1_wt_delta.T * alpha
    
    
    y_axis_error[0,epoch] = error
    
    if(epoch % 10 == 0):
         print("error", error)
         print("actual", actual)
         print("pred  ", l2)


plt.title("Error over epochs")
plt.xlabel("epochs")
plt.ylabel("error")
plt.plot(x_axis_epochs.ravel(), y_axis_error.ravel(), color ="blue")
plt.show()

    
    
