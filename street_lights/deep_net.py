import numpy as np

np.random.seed(1)

def relu(x):

    return (x > 0) * x


def relu_derivative(x):

    return (x > 0)


# Input
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])  # (6,3)

# Known Data
walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

# Knobs
layer_1_hidden_nodes = 4
alpha = 0.2
epochs = 60

# Weights
weights_0_1 = 2*np.random.random((3, layer_1_hidden_nodes))-1 # (3,4)
weights_1_2 = 2*np.random.random((layer_1_hidden_nodes,1))-1 # (4, 1)


for epoch in range(epochs):

    total_error = 0

    for input, actual in zip(streetlights, walk_vs_stop):

        print("input shape = " + str(input.shape)) # (1,3)
        
        # Predict
        layer_1 = relu(input.T.dot(weights_0_1)) # (4,1)
        layer_2 = layer_1.dot(weights_1_2) # ()

        print("layer_1 shape = " + str(layer_1.shape))
        print("layer_2 shape = " + str(layer_2.shape))








