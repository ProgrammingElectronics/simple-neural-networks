import numpy as np
import matplotlib.pyplot as plt

weight = 0.5
actual = 0.8
input = 2
table = np.zeros((20,2))

for iteration in range(20):
    #Predict
    pred = input * weight
    
    #Compare | Compute Error
    error = (pred - actual) ** 2
    table[iteration] = [pred, error]
    
    delta = pred - actual
    weight_delta = delta * input

    #Adjust Weights
    weight -= weight_delta

    print("Error:" + str(error) + "Prediction:" + str(pred))


plt.plot(table[:,0], table[:,1])
plt.show()/watch