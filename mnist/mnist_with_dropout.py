import numpy as np
import matplotlib.pyplot as plt

### Functions

# Create Embedding
def create_embedding(x):
    embedding = np.zeros(10)
    embedding[x] = 1
    return embedding

# Relu
def relu(x):

    return (x >= 0) * x

# Relu Derivative
def relu_deriv(x):

    return (x >= 0)

# Set Random
np.random.seed(1)

# Import Data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Inspect Data
assert x_train.shape == (60000,28,28) , f"expected (60000,24,24), actual {x_train.shape}"
assert y_train.shape == (60000,) , f"expected (60000,), actual {y_train.shape}"

assert x_test.shape == (10000,28,28) , f"expected (10000,24,24), actual {x_train.shape}"
assert y_test.shape == (10000,) , f"expected (10000,), actual {y_train.shape}"

# Form Data
images = x_train.reshape((60000,784)) / 255
labels = np.array([create_embedding(y) for y in y_train])

test_images = x_test.reshape((10000,784)) / 255
test_labels = np.array([create_embedding(y) for y in y_test])

assert images.shape == (60000,784) , f"Expected (60000,728), was {images.shape}"
assert labels.shape == (60000,10) , f"Expected (60000,10), was {labels.shape}"

assert test_images.shape == (10000,784) , f"Expected (10000,728), was {images.shape}"
assert test_labels.shape == (10000,10) , f"Expected (10000,10), was {labels.shape}"

assert np.argmax(test_labels[0:1]) == y_test[0:1], f"{test_labels[0:1]} != {y_test[0:1]}" 

# Architechture
hidden_layer_size = 100

# Weights
w_0_1 = 0.2*np.random.random((784,hidden_layer_size)) - 0.1
w_1_2 = 0.2*np.random.random((hidden_layer_size,10)) - 0.1

# Controls
epochs = 300
batch_size = 100
alpha = 0.001
sample_size = 1000

# Visualize
plot_error = np.zeros(epochs)

### Train
for epoch in range(epochs):

    error = 0.0
    correct_cnt = 0

    for input, label in zip(images[:sample_size], labels[:sample_size]):
    
        input = input.reshape((1,784))
        label = label.reshape((1,10))

        assert input.shape == (1,784) , f"Expected (1,784), was {input.shape}"
        assert label.shape == (1,10) , f"Expected (1,10), was {label.shape}"
        

        ### Predict with drop out
        l1 = relu(input.dot(w_0_1))
        dropout_mask = np.random.randint(2, size=l1.shape)
        l1 *= dropout_mask * 2
        l2 = l1.dot(w_1_2)

        ### Compare
        error += np.sum((label - l2) ** 2)
        correct_cnt += int(np.argmax(l2) == np.argmax(label))

        # Delta
        l2_delta = l2 - label
        l1_delta = l2_delta.dot(w_1_2.T)*relu_deriv(l1) # Back propogation
        l1_delta *= dropout_mask

        ### Update Weights

        # Wt Delta
        l2_wt_delta = l2_delta.T.dot(l1)
        l1_wt_delta = l1_delta.T.dot(input)

        # Adjust Weights
        w_1_2 -= l2_wt_delta.T * alpha
        w_0_1 -= l1_wt_delta.T * alpha

    # Print Error
    print("Epoch", epoch, "Error", error/float(sample_size), "Correct", correct_cnt / float(sample_size))
    plot_error[epoch] = error

### TEST ##########

error = 0
correct_cnt = 0

test_sample_size = len(x_test)
assert test_sample_size == 10000 , f"expect 10000, is {test_sample_size}"

for image, label in zip(test_images[0:test_sample_size], test_labels[0:test_sample_size]):

    # Form the data
    l0 = image.reshape((1,784))
    label = label.reshape((1,10))

    # Predict
    l1 = relu(l0.dot(w_0_1))
    l2 = l1.dot(w_1_2)

    # Compare
    error += np.sum((l2 - label) ** 2)
    correct_cnt += int(np.argmax(l2) == np.argmax(label))

print("Test Error", error/float(test_sample_size), "Test Correct", correct_cnt/float(test_sample_size))

# Plot Error
fig, ax = plt.subplots()
ax.plot(range(epochs), plot_error)
plt.show()


