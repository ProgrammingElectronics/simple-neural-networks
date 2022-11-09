import numpy as np

a = np.array([0,1,2,3])
b = np.array([4,5,6,7])
c = np.array([[0,1,2,3],
              [4,5,6,7]])

d = np.zeros((2,4))
e = np.random.rand(2,5)

print(a)
print(b)
print(c)
print(d)
print(e)


print(a * 0.1)
print(c * 0.2)
print(a * b)
print(a * b * 0.2)
print(a * c)

print(a * e)
