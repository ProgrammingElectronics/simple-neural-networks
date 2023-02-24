import numpy as np

a = np.array([(1,1,1),(2,2,2),(3,3,3)])
b = np.array([(4,4,4),(5,5,5),(6,6,6)])



assert a.shape == (3,3) , f"Expected (3,3) actual {a.shape}"
assert b.shape == (3,3) , f"Expected (3,3) actual {b.shape}"


