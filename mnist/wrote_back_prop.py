#############################################
# L_0    ## (784,1)         -> input
# WT_0_1 ## (784,100)
# L_1    ## (100,1)         -> hidden layer 1
# WT_1_2 ## (100,50)
# L_2    ## (50,1)          -> hidden layer 2
# WT_2_3 ## (50, 10)
# L_3    ## (10,1)          -> output
############################################

### imports
import numpy as np
import matplotlib.pyplot as plt

### import dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

### inspect dataset
print("x_train", x_train.shape)
print("y_train",y_train.shape)

### create levers


### create weights

### predict

### compare

### adjust weights ###

# calculate delta for each layer
# calculate wt_delta for each layer
# adjust weights 

