import numpy as np

def neural_network(input, weights, actual, alpha):

    pass



# Weights
weights = np.array([0.3, 0.2, 0.9])

# Data
hurt = np.array([0.1, 0.0, 0.0, 0.1])
win = np.array([1, 1, 0, 1])
sad = np.array([0.1, 0.0, 0.1, 0.2])

wlrec = np.array([0.65, 1.0, 1.0, 0.9])

actual = np.array([hurt[0], win[0], sad[0]])
input = np.array(wlrec[0])

# Knobs
epochs = 200
alpha = 0.1


for epoch in range(epochs):

    # Predict
    pred = input * weights # 1 by 3 array

    # Compare
    mse = (pred - actual) ** 2  
    delta = pred - actual  

    # Adjust
    weight_delta = delta * input 
    weights -= weight_delta * alpha

    print("Error: " + str(mse) + "Prediction: " + str(pred))