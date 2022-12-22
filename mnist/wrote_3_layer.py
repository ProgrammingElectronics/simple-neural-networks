# imports
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# functions
def createEmbedding(val):
    embedding = np.zeros(10)
    embedding[val] = 1
    return embedding

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
weights_0_1 = np.random.random((784, l1_nodes))
weights_1_2 = np.random.random((l1_nodes, 10))

# trainging controls
epochs = 10
alphas = np.zeros(10)
alphas[0] = 0.001
for i in range(1,10):    
    alphas[i] = alphas[i-1] * 0.1
    
# output array for graph
errors = np.zeros(len(alphas))

for epoch in range(epochs):
        
    for input, label in zip (inputs, labels):

        pass




plt.title("erres with different alphas")
plt.xlabel("error")
plt.ylabel("epochs")
plt.plot(errors, range(epochs))
plt.show()

