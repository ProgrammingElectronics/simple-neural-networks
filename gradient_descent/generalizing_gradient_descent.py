import numpy as np

def neural_network(input, weights, alpha):
    # Predict -> dot product
    pred = np.dot(input,weights)

    # Compare
    mse = (pred - actual) ** 2
    print("Prediction: " + str(pred) + " MSE: " + str(mse))
    delta = pred - actual
    weight_delta = delta * input # elementwise scalar multiplication

    # Adjust weights
    weights -= weight_delta * alpha
    print("Weights: " + str(weights))
    print(" Weight Deltas: " + str(weight_delta))

# Weights
weights = np.array([0.1, 0.2, -0.1])

# Data
toes = 8.5
win_loss = 0.65
fans = 1.2
input = np.array([toes, win_loss, fans])
actual = 1

# paramaters
alpha = 0.01
epocs = 20

for iteration in range(epocs):
    neural_network(input, weights, alpha)










