import numpy as np
import matplotlib.pyplot as plt

def neural_network(input, weight):
    return input * weight

def mean_squared_error(pred, actual):
    return (pred - actual) ** 2

weight = 0.1
alpha = 0.01

number_of_toes = [8.5]
win_or_lose_binary = [1] # win yo!

input = number_of_toes[0]
actual = win_or_lose_binary[0] 

table = np.array([[],[]])

for iteration in range(20):

    # Predict
    pred = neural_network(input, weight)

    # Compare
    error = mean_squared_error(pred, actual)

    table[iteration][0] = weight
    table[iteration][1] = error
    
    # print("error: ", error, " pred: ", pred)
    # Learn
    delta = pred - actual
    wt_delta = delta * input
    weight -= wt_delta * alpha


print(table)