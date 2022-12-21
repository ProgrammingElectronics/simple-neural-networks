#############################################
# L_0    ## (784,1)         -> input
# WT_0_1 ## (784,100)
# L_1    ## (100,1)         -> hidden layer 1
# WT_1_2 ## (100,50)
# L_2    ## (50,1)          -> hidden layer 2
# WT_2_3 ## (50, 10)
# L_3    ## (10,1)          -> output
############################################

### imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def createEmbedding(val):

    embedding = np.zeros((10))
    embedding[val] = 1
    return embedding

def relu(x):

    return (x > 0) * x

def relu_1st_deriv(x):

    return (x > 0)

#  Hmmmmm
def relu_2nd_deriv(x):

    return 0


### Print flags
print_preamble = True
print_error = True
print_freq = 1

### import dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

### inspect dataset
if(print_preamble):
    print("x_train", x_train.shape)
    print("y_train",y_train.shape)

### prep data
inputs = x_train.reshape((len(x_train), 784))
labels = np.array([createEmbedding(y) for y in y_train])

if(print_preamble):
    print("input shape", inputs.shape)
    print("actual shape", labels.shape)

### Network Arch
hl_1_nodes = 100
hl_2_nodes = 100

### Network Controls
alpha = 0.000000000000000001
epochs = 100

### create weights
wt_0_1 = np.random.random((784, hl_1_nodes))
wt_1_2 = np.random.random((hl_1_nodes, hl_2_nodes))
wt_2_3 = np.random.random((hl_2_nodes, 10))


for epoch in range(epochs):

    ### print flags
    print_once = True
    
    error = 0

    for input, actual in zip(inputs[0:1000], labels[0:1000]):

        ### reshape input and actual
        l_0 = input.reshape(1,784)
        actual = actual.reshape(1,10)

        if(print_error and epoch % print_freq == 0 and print_once):
            print("##############")
            print("epoch", epoch)
            print("input_shape", l_0.shape)
            print("actual_shape", actual.shape)

        ### predict
        l_1 = l_0.dot(wt_0_1)
        l_2 = relu(l_1.dot(wt_1_2))
        l_3 = relu(l_2.dot(wt_2_3))
        
        if(print_error and epoch % print_freq == 0 and print_once):
            print("l_1.shape", l_1.shape)
            print("l_2.shape", l_2.shape)
            print("l_3.shape", l_3.shape)

        ### compare
        error += np.sum((l_3 - actual)**2)
        
        if(print_error and epoch % print_freq == 0 and print_once):
            print("error------->", error)
        
        ### adjust weights ###
        
        # calculate delta for each layer
        l_3_delta = l_3 - actual
        l_2_delta = l_3_delta.dot(wt_2_3.T) * relu_1st_deriv(l_2)
        l_1_delta = l_2_delta.dot(wt_1_2.T) * relu_1st_deriv(l_1)

        if(print_error and epoch % print_freq == 0 and print_once):
                    print("l_3_delta.shape", l_3_delta.shape)
                    print("l_2_delta.shape", l_2_delta.shape)
                    print("l_1_delta.shape", l_1_delta.shape)

        # calculate wt_delta for each layer and adjust weights
        wt_2_3 -= l_2.T.dot(l_3_delta) * alpha #   l_3_delta.T.dot(l_2) * alpha
        wt_1_2 -= l_1.T.dot(l_2_delta) * alpha #   l_2_delta.T.dot(l_1) * alpha
        wt_0_1 -= l_0.T.dot(l_1_delta) * alpha #   l_1_delta.T.dot(l_0) * alpha

        if(print_error and epoch % print_freq == 0 and print_once):
            print("actual", actual)
            print("pred  ", l_3)

        print_once = False
