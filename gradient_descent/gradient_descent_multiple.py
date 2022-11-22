import numpy as np

# Weights
weights = np.array([
    [0.1, 0.1, -0.3],  # hurt?
    [0.1, 0.2, 0.0],  # win?
    [0.0, 1.3, 0.1]])  # sad?

# data
toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])  # Win-loss record
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input = np.array([toes[0], wlrec[0], nfans[0]])

# actual
hurt = np.array([0.1, 0.0, 0.0, 0.1])
win = np.array([1, 1, 0, 1])
sad = np.array([0.1, 0.0, 0.1, 0.2])

actual = np.array([hurt[0], win[0], sad[0]])

# Knobs
epochs = 200
alpha = 0.01


for epoch in range(epochs):

    # Predict
    pred = input.dot(weights) # Pred is a 1,3 array
    
    # Compare
    mse = (pred - actual) ** 2
    delta = pred - actual

    # Adjust
    weight_delta = delta * input
    weights -= weight_delta * alpha

    print("Error: " + str(mse) + "Prediction: " + str(pred))