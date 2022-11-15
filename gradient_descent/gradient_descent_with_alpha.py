import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Data
weight = 0.5
init_input = 2
actual = 0.8
init_alpha = .1
epochs = 20


def neural_network(input, weight, alpha, epochs):
    
    table = np.zeros((epochs,2)) #  (weight, mse)

    for iteration in range(epochs):

        # Predict
        pred = input * weight

        # Compare / Compute Error
        mse = (pred - actual) ** 2
        table[iteration] = (weight, mse)
        
        delta = pred - actual
        weight_delta = delta * input

        # Adjust Weights
        weight -= weight_delta * alpha

        print("Error:" + str(mse) + " Prediction:" + str(pred))

    return table


fig, ax = plt.subplots()
ax.plot(neural_network(init_input, weight, init_alpha, epochs)[:,0], neural_network(init_input, weight, init_alpha, epochs)[:,1])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Slider for Alpha
ax_alpha = fig.add_axes([0.25, 0.1, 0.65, 0.03])
alpha_slider = Slider(
    ax = ax_alpha,
    label = "alpha",
    valmin = 0,
    valmax= 1,
    valinit=init_alpha
)

# Slider for INPUT
ax_input = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
input_slider = Slider(
    ax = ax_input,
    label = "input",
    valmin = -20,
    valmax= 20,
    valinit=init_input,
    orientation="vertical"
)

# Update function
def update(val):
    ax.clear()
    ax.plot(neural_network(input_slider.val, weight, alpha_slider.val, epochs)[:,0], neural_network(input_slider.val, weight, alpha_slider.val, epochs)[:,1])
    print(neural_network(input_slider.val,weight,alpha_slider.val, epochs))


alpha_slider.on_changed(update)
input_slider.on_changed(update)

plt.show()

# plt.ylabel("Error")
# plt.xlabel("Weight")

# plt.plot(data[:,0], data[:,1])