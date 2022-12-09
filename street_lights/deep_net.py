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
epochs = 1

# Weights
weights_0_1 = 2*np.random.random((3, layer_1_hidden_nodes))-1  # (3,4)
weights_1_2 = 2*np.random.random((layer_1_hidden_nodes, 1))-1  # (4, 1)


for epoch in range(epochs):

    total_error = 0

    for input, actual in zip(streetlights, walk_vs_stop):

        # Predict
        layer_1 = relu(input.dot(weights_0_1))  # (3,1) dot (3,4) -> (4,1)
        layer_2 = layer_1.dot(weights_1_2)  # (4,1) dot (4,1) -> (1,1)
        
        # Compare
        mse = (layer_2 - actual) ** 2
        print(mse)

        # Learn -> Adjust Weights
        
        # Calculate weight deltas for each layer, start at output and working backwards
        layer_2_delta = layer_2 - actual
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu_derivative(layer_1)  # (1,1) dot (1,4) -> (1,4) * (4,1)
        
        #layer_2_wt_delta = layer_1.T.dot(layer_2_delta)  # Scalar * (1,1)


        # layer 1 to layer 0
        

