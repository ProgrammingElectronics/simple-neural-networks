import numpy as np

# Import mnist data set
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.random.seed(42)


def createEmbedding(val):

    embedding = np.zeros((1,10))
    embedding[0,val] = 1
    return embedding

# Inspect Data
print("input shape = ", x_train.shape)
print("label shape = ", y_train.shape)

# Prepare data
sample_input = x_train[0].ravel().reshape((1,784))
sample_label = y_train[0]
sample_embedding = createEmbedding(sample_label)
print("label", sample_label)
print("embedding", sample_embedding, sample_embedding.shape)

# create activation RELU function for prediction
def relu(x):

    return (x > 0) * x

# create derivative RELU function for back-propobation
def relu(x):

    return (x > 0)


# Create knobs
hidden_layer_nodes = 100
alpha = 0.1
print_shape = True

# Create weight arrays for WT_0_1 and WT_1_2
WT_0_1 = 2*np.random.random((sample_input.size,hidden_layer_nodes))-1 # (784,100)
if(print_shape):
    print("WT_0_1", WT_0_1.shape)


# Predict L1 and L2

# Compare -> calculate MSE

# Calculate delta for l2 and l1
    # L2_delta = pred - actual
    # L1_delta = L2_delta.dot(weights_1_2) * ReluDeriv(layer_1)

# Calculate WT_delta for L2 and L1
    # L2_WT_delta = L2_delta.dot(layer_1)
    # L1_WT_delta = L1_delta.dot(layer_0)

# Adjust weights with weight delta
    # weights_1_2 -= L2_WT_delta * alpha
    # weights_0_1 -= L1_WT_delta * alpha




