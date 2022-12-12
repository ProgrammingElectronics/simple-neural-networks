import numpy as np

###################################
#
# (input)                   (hidden)                  (output)
# layer_0 -> weights_0_1 -> layer_1 -> weights_1_2 -> layer_2
#  (3,1)       (3,4)         (3,4)      (1,1)
#
###################################

np.random.seed(1)


def displayShape(label,matrix):

    print(label, str(matrix.shape) )


def displayDotCalc(eq_label, matrix_l, matrix_r, answer_label):
    print()
    print(eq_label, str(matrix_l.shape), "dot", str(matrix_r.shape), "->",str(answer_label.shape))


def relu(x):

    return (x > 0) * x


def relu_derivative(x):

    return (x > 0)


# Input
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1],
#                          [0, 1, 1],
#                          [1, 0, 1]])  # (6,3)

streetlights = np.array( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )

# Known Data
# walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
walk_vs_stop = np.array([1, 1, 0, 0])

# Knobs
layer_1_hidden_nodes = 4
alpha = 0.2
epochs = 60
show_shapes = False

# Weights
weights_0_1 = 2*np.random.random((3, layer_1_hidden_nodes))-1  # (3,4)
weights_1_2 = 2*np.random.random((layer_1_hidden_nodes, 1))-1  # (4, 1)

if(show_shapes):
    displayShape("weights_0_1", weights_0_1)
    displayShape("weights_1_2", weights_1_2)


for epoch in range(epochs):

    total_error = 0

    for input, actual in zip(streetlights, walk_vs_stop):

        ### Predict ###
        layer_0 = input.reshape(1,3)
        layer_1 = relu(layer_0.dot(weights_0_1))  # (1,3) dot (3,4) -> (1,4)
        layer_2 = layer_1.dot(weights_1_2)  # (1,4) dot (4,1) -> (1,1)
        
        if(show_shapes):
            displayDotCalc("layer_0 dot weights_0_1", layer_0, weights_0_1, layer_1)
            displayDotCalc("layer_1 dot weights_1_2", layer_1, weights_1_2, layer_2)
            
        ### Compare ###
        mse = (layer_2 - actual) ** 2
        # total_error += np.sum(mse)
        total_error += mse

        ### Learn ###
        
        # Calculate deltas for each layer, start at output and work backwards
        layer_2_delta = layer_2 - actual
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu_derivative(layer_1)  # (1,1) dot (1,4) -> (1,4) * (4,1)
        
        if(show_shapes):
            displayDotCalc("layer_2_delta dot weights_1_2", layer_2_delta, weights_1_2.T, layer_1_delta)

        # Calculate Weight Deltas
        layer_2_wt_delta = layer_2_delta.dot(layer_1) # (1,1) dot (1,4) -> (1,4)
        layer_1_wt_delta = layer_1_delta.T.dot(layer_0) # (4,1) dot (1,3) -> (4,3)

        if(show_shapes):
            displayDotCalc("layer_2_delta dot layer_1", layer_2_delta, layer_1, layer_2_wt_delta)
            displayDotCalc("layer_1_delta.T dot layer_0", layer_1_delta.T, layer_0, layer_1_wt_delta)

        # Adjust Weights
        weights_1_2 -= layer_2_wt_delta.T * alpha
        weights_0_1 -= layer_1_wt_delta.T * alpha
    
    if(epoch % 10 == 9):
        print("total error", total_error)


