import numpy as np
import matplotlib.pyplot as plt

def neural_network(input, weight):
    return input * weight

def mean_squared_error(pred, actual):
    return (pred - actual) ** 2

weight = 0.01
alpha = 0.01

number_of_toes = [8.5]
win_or_lose_binary = [1] # win yo!

input = number_of_toes[0]
actual = win_or_lose_binary[0] 

table = np.zeros((20,2))

for iteration in range(20):

    # Predict
    pred = neural_network(input, weight)

    # Compare
    error = mean_squared_error(pred, actual)

    table[iteration] = [weight, error]
    
    # Learn
    delta = pred - actual
    wt_delta = delta * input
    weight -= wt_delta * alpha

print(table)
plt.xlabel("weight")
plt.ylabel("error")
plt.plot(table[:,0], table[:,1])
plt.show()
