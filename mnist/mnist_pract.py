import numpy as np
import matplotlib.pyplot as plt

### Functions

# Create Embedding
def create_embedding(x):
    embedding = np.zeros(10)
    embedding[x] = 1
    return embedding

# Relu
def relu(x):

    return (x > 0) * x

# Relu Derivative
def relu_deriv(x):

    return (x >= 0)

# Set Random
np.random.seed(1)

# Import Data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Inspect Data
assert x_train.shape == (60000,28,28) , f"expected (60000,24,24), actual {x_train.shape}"
assert y_train.shape == (60000,) , f"expected (60000,), actual {y_train.shape}"

# Form Data
inputs = x_train.reshape((60000,784)) / 255
labels = np.array([create_embedding(y) for y in y_train])

assert inputs.shape == (60000,784) , f"Expected (60000,728), was {inputs.shape}"
assert labels.shape == (60000,10) , f"Expected (60000,10), was {labels.shape}"

# Architechture
hidden_layer_size = 40

# Weights
w_0_1 = 0.2*np.random.random((784,hidden_layer_size)) - 0.1
w_1_2 = 0.2*np.random.random((hidden_layer_size,10)) - 0.1

# Controls
epochs = 350
alpha = 0.005
sample_size = 1000

# Visualize
plot_error = np.zeros(epochs)

### Train
for epoch in range(epochs):

    total_error = 0

    for input, label in zip(inputs[:sample_size], labels[:sample_size]):
    
        input = input.reshape((1,784))
        label = label.reshape((1,10))

        assert input.shape == (1,784) , f"Expected (1,784), was {input.shape}"
        assert label.shape == (1,10) , f"Expected (1,10), was {label.shape}"
        
        ### Predict
        l1 = relu(input.dot(w_0_1))
        l2 = l1.dot(w_1_2)

        ### Compare
        error = (l2 - label) ** 2
        total_error += np.sum(error)

        # Delta
        l2_delta = l2 - label
        l1_delta = l2_delta.dot(w_1_2.T)*relu_deriv(l1) # Back propogation

        ### Update Weights

        # Wt Delta
        l2_wt_delta = l2_delta.T.dot(l1)
        l1_wt_delta = l1_delta.T.dot(input)

        # Adjust Weights
        w_1_2 -= l2_wt_delta.T * alpha
        w_0_1 -= l1_wt_delta.T * alpha

    # Print Error
    print("total error", total_error)
    plot_error[epoch] = total_error

# Plot Error
fig, ax = plt.subplots()
ax.plot(range(epochs), plot_error)
plt.show()
