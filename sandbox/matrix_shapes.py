import numpy as np

def m_display(array):
    print(array.shape)
    print(array)

raw = np.array([[0, 1, 0],
              [0, 1, 0],
              [0, 1, 1]])

m_display(raw)

input = raw.reshape((9, 1))
m_display(input.T)

output = np.zeros((2,1))
m_display(output)

weights = np.zeros((9,2))
weights[1,0] = 2
weights[5,0] = 2
weights[2,1] = 1
weights[5,1] = 3
weights[7,1] = 1

m_display(weights)

dot_product = input.T.dot(weights)
m_display(dot_product)