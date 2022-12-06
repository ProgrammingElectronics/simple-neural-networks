from keras.datasets import mnist
import numpy as np

def inspect_array_to_console(array, label = "info"):

    print("########### " + label + " #################")
    print("shape: ", array.shape)
    print("nDim: " , array.ndim)
    print("Length: ", len(array))
    print("Size: ", array.size)
    print(array[0:1])
    print("")

def create_embedding(val):
    embedding = np.zeros((10))
    embedding[val] = 1
    result = np.reshape(embedding, (1,10))
    return result

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Get single data point
images = x_train [0:1]
labels = y_train[0:1]
#inspect_array_to_console(images, "Raw images")
#inspect_array_to_console(labels, "Labels")

# Get just first image
image = images[0]
#inspect_array_to_console(image, "Single images")

# Concatenate image into a 1 by 784 len array
flat_image = np.reshape(image, (1,784)) # (784,1)
inspect_array_to_console(flat_image, "Flat image")

# Data
input = flat_image
actual = create_embedding(labels) # (1,10)
inspect_array_to_console(actual, "actual") 

# Weights
# weights = np.random.random((784,10))
# weights = np.zeros((784,10))
weights = np.random.random((784,10))
inspect_array_to_console(weights, "weights")

# Knobs
epochs = 20
alpha = 0.0000001

for epoch in range(epochs):

    # Predict
    pred = input.dot(weights) # (1, 784) dot (784,10) -> (1,10)
    #inspect_array_to_console(pred, "pred")
    
    # Compare
    mse = (pred - actual) ** 2 # -> (1,10)
    #inspect_array_to_console(mse, "mse")
    delta = pred - actual # -> (1, 10)
    #inspect_array_to_console(delta, "delta")
    
    # Adjust Weights
    weight_delta = delta.T * input # (1,10) * (1,784) -> (10, 784) 
    #inspect_array_to_console(weight_delta, "weight_delta") 

    weights -= weight_delta.T * alpha # (784,10) - (10,784) -> (784,10)
    #inspect_array_to_console(weights, "weights") 
    
    print("Error" + str(mse))
    # print("Actual" + str(actual))
    print("Pred" + str(pred))



#########################  SAND BOXING  #################################
# a = np.array([
#         [0,1,2],
#         [3,4,5]
# ])
# inspect_array_to_console(a, "a") 

# c = np.reshape(a, (6,1))


# # inspect_array_to_console(a.ravel(), "a.ravel")
# inspect_array_to_console(c, "c") 

# b = np.ones((7,1))
# inspect_array_to_console(b, "b") 



