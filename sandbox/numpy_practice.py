# import numpy as np
# import matplotlib.pyplot as plt

# a_range = np.arange(0,2).reshape((1, 2))
# lin_space = np.arange(0,2).reshape((1, 2))

# print("arange", a_range, a_range.shape)
# print("lin_space", lin_space, lin_space.shape)



# plt.title("Error over epochs")
# plt.xlabel("epochs")
# plt.ylabel("error")
# plt.plot(a_range, lin_space)
# plt.show()



# importing the modules
import numpy as np
import matplotlib.pyplot as plt
 
# data to be plotted
x = np.arange(1, 11)
y = x * x
 
# plotting
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="blue")
plt.show()