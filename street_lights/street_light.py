import numpy as np

# Input
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])  # (6,3)

# Known Data
# walk_vs_stop = np.array([[0],
#                          [1],
#                          [0],
#                          [1],
#                          [1],
#                          [0]])  # (6,1)
walk_vs_stop = np.array( [ 0, 1, 0, 1, 1, 0 ] )


# Actual
# input_data = streetlights # (1,3)

# actual = walk_vs_stop[0][0]   # (1,1)

# Weights
# Rows -> Number of "features" in input | Columns -> Number of nodes in output
# rng = np.random.default_rng(seed=42)
# weights = rng.random((3))
weights = np.array([0.5,0.48,-0.7])

# Knobs
epochs = 40
alpha = 0.1

for epoch in range(epochs):

    error_for_all_lights = 0
    for input, actual in zip(streetlights, walk_vs_stop):
        
        # Predict
        pred = input.dot(weights)
        
        # Compare
        mse = (pred - actual) ** 2
        error_for_all_lights += mse
        delta = pred - actual
        
        # Learn AKA update weights
        weight_delta = delta * input
        weights -= weight_delta * alpha

    print("prediction " + str(pred))
    print("error for all " + str(error_for_all_lights) + "\n")

print(weights)
