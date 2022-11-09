import numpy as np
import matplotlib.pyplot as plt

input = 0.5
actual = 0.8
weight = np.arange(-5,7,0.01)
error = ((input * weight) - actual) ** 2

plt.xlabel("weight")
plt.ylabel("error")
plt.plot(weight, error)
plt.show()



