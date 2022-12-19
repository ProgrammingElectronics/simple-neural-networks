#############################################
# L_0    ## (784,1)         -> input
# WT_0_1 ## (784,100)
# L_1    ## (100,1)         -> hidden layer
# WT_1_2 ## (100,10)
# L_2    ## (10,1)          -> output
############################################

import numpy as np

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
hidden_layer_nodes = 10
output_layer_nodes = 10

### Create knobs ###
alpha = 0.000000001
print_shape = True
epochs = 60

### Inspect Data ###
# if (print_shape):
#     print("inputs shape raw = ", x_train.shape)
#     print("labels shape raw = ", y_train.shape)

## Prepare data ###
inputs = x_train.reshape((60000, 784))

# this is what I want to do -> [createEmbedding(label) for label in y_train]
labels = np.zeros((60000, 10))
for i, label in enumerate(y_train):
    labels[i] = createEmbedding(label)

# if (print_shape):
#     print("inputs shape new = ", inputs.shape)
#     print("labels shape new", labels.shape)

### Create weight arrays for WT_0_1 and WT_1_2 ###
wt_0_1 = 2*np.random.random((input_layer_nodes,
                            hidden_layer_nodes))-1  # (784,100)
wt_1_2 = 2*np.random.random((hidden_layer_nodes,
                            output_layer_nodes))-1  # (100,10)

# if (print_shape):
#     print("WT_0_1", wt_0_1.shape)
#     print("WT_1_2", wt_1_2.shape)


for epoch in range(epochs):
    
    print_flag = True

    error = 0

    for input, label in zip(inputs[0:100], labels[0:100]):

        # Prep input and label
        l0 = input.reshape((1, 784))
        actual = label.reshape((1, 10))

        # if (print_shape and print_flag):
        #     print("input shape =", l0.shape)  # (1, 784)
        #     print("l0 sum= ", l0.sum())
        #     print("label shape =", actual.shape)  # (1,10)
        #     print("actual = ", actual)

        ### Predict L1 and L2
        l1 = relu(l0.dot(wt_0_1))  # (1,784) dot (784,100) -> (1,100)
        l2 = l1.dot(wt_1_2) # (1,100) dot (100, 10) -> (1,10)

        # if (print_shape and print_flag):
        #     print("l1 shape =", l1.shape)  # (1, 784)
        #     print("l2 shape =", l2.shape)  # (1,10)

        #### Compare -> calculate MSE
        error = np.sum((l2 - actual) ** 2)
        
        if (print_shape and print_flag):
            print("######################")
            print("l2 =", l2)
            print("actual =", actual)
            print("##")
            print("error =", error)
            
        # Calculate delta for l2 and l1
        l2_delta = l2 - actual # (1,10) - (1, 10) -> (1, 10)
        l1_delta = l2_delta.dot(wt_1_2.T) * relu_deriv(l1) # (1, 10) dot (10, 100) -> (1,100) 
    
        # if (print_shape and print_flag):
        #     print("l2_delta shape =", l2_delta.shape)
        #     print("l1_delta shape =", l1_delta.shape)

        # Calulate weight delta
        # l2_wt_delta = l2_delta.T.dot(l1) # (10,1) dot (1,100) -> (10, 100)
        # l1_wt_delta = l1_delta.T.dot(l0) # (100,1) dot (1,784) -> (100, 784)
    
        # if (print_shape and print_flag):
        #     print("l2_wt_delta shape =", l2_wt_delta.shape)
        #     print("l1_wt_delta shape =", l1_wt_delta.shape)

        # Adjust weights
        # wt_1_2 -= l2_wt_delta.T * alpha
        # wt_0_1 -= l1_wt_delta.T * alpha
        wt_1_2 -= alpha * l1.T.dot(l2_delta)
        wt_0_1 -= alpha * l0.T.dot(l1_delta) 

        # print_flag = False
    
    # if(epoch % 3 == 0):
    #     print("total_error", total_error)
    
    
