# imports
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# functions
def createEmbedding(val):
    embedding = np.zeros(10)
    embedding[val] = 1
    return embedding

def relu(x):

    return (x > 0) * x

def relu_1st_deriv(x):

    return (x > 0)

# set random
np.random.seed(42)

# import data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# confirm data
assert x_train.shape == (60000,28,28), f"Images should be (60000,28,28), but are {x_train.shape}"
assert y_train.shape == (60000,), f"Labels should be (60000,) but is {y_train.shape}"

# transform data
inputs = x_train.reshape((60000, 784))
labels = np.array([createEmbedding(y) for y in y_train]) 

assert inputs.shape == (60000,784), f"inputs shape should be (60000,784), but is {inputs.shape}"
assert labels.shape == (60000,10), f"labels shape should be (60000, 10), but is {labels.shape}"

# network
l1_nodes = 100
weights_0_1 = 2*np.random.random((784, l1_nodes))-1
weights_1_2 = 2*np.random.random((l1_nodes, 10))-1

# trainging controls
epochs = 10
start_alpha = 9
end_alpha = start_alpha + 10
alphas = np.power(10., -np.arange(start_alpha,end_alpha))
# alphas = np.zeros(10)
# alphas[0] = 0.00000000001
# for i in range(1,10):    
#     alphas[i] = alphas[i-1] * 0.1

print(alphas)
    
# output array for graph
errors = np.zeros((epochs))
assert len(errors) == epochs, f"errors array should number of epochs, but is instead {len(errors)}"

for alpha_index, alpha in enumerate(alphas):

    print("########### alpha", alpha)

    for epoch in range(epochs):
            
        errors[epoch] = 0
        
        for input, label in zip (inputs[0:10], labels[0:10]):

            # prep input
            l0 = input.reshape((1,784))
            actual = label.reshape((1,10))
            assert l0.shape == (1,784), f"l0 shape should be (1,784), but is {l0.shape}"
            assert actual.shape == (1,10), f"label shape should be (1,10), but is {actual.shape}"

            # Predict
            l1 = relu(l0.dot(weights_0_1))
            l2 = l1.dot(weights_1_2)

            assert l1.shape == (1,100), f"l1 chape should be (1,100), but is {l1.shape}" 
            assert l2.shape == (1,10), f"l2 shape should be (1,10), but is {l2.shape}"

            # Compare
            errors[epoch] += np.sum((l2 - actual) ** 2)
            assert not np.isnan(errors[epoch]), f"error in epoch {epoch} has gone NaN" 

            l2_delta = l2 - actual
            l1_delta = l2_delta.dot(weights_1_2.T) * relu_1st_deriv(l1)

            assert l2_delta.shape == (1,10), f"shape of l2_delta should be (1,10), but is {l2_delta.shape}"
            assert l1_delta.shape == (1,100), f"shape of l1_delta should be (1,100), but is {l1_delta.shape}"

            # adjust weights
            l2_wt_delta = l2_delta.T.dot(l1)
            l1_wt_delta = l1_delta.T.dot(l0)

            assert l2_wt_delta.shape == (10,100), f"l2_wt_delta shape should be (10,100), but it {l2_wt_delta.shape}"
            assert l1_wt_delta.shape == (100,784), f"l2_wt_delta shape should be (10,100), but it {l1_wt_delta.shape}"

            weights_1_2 -= l2_wt_delta.T * alpha
            weights_0_1 -= l1_wt_delta.T * alpha

        print("Epoch", epoch, "Error", errors[epoch])

    plt.title("erros w/ different alphas")
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.plot(range(epochs), errors, label=alpha)
    plt.legend()    

plt.show()

# Combined Graph
    # plot_title = f"error with alpha = {alpha}"
    # fig, axs = plt.subplots(len(alphas), sharex=True)
    # axs[alpha_index].set_title(plot_title) 
    # axs[alpha_index].set_ylabel("error")
    # axs[alpha_index].plot(range(epochs), errors)

