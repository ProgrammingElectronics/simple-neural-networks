import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def f(input, weight, actual):

    for iteration in range(20):
        #Predict
        pred = input * weight
        
        #Compare
        delta = pred - actual
        weight_delta = delta * input

        #Adjust
        weight -= weight_delta

    return weight





