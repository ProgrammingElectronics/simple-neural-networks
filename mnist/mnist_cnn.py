## imports #################
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

np.random.seed(1)

## Functions #################

# one hot embedding


def embedding(x):

    embedding = np.zeros((1, 10))
    embedding[0, x] = 1
    return embedding

# tanh


def tanh(x):

    return np.tanh(x)

# tanh derivative


def tanh_deriv(x):

    return 1 - (x ** 2)

# softmax


def softmax(x):

    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

# convolution


def get_image_section(layer, start_row, stop_row, start_col, stop_col):

    section = layer[:, start_row:stop_row, start_col:stop_col]
    return section.reshape(-1, 1, stop_row-start_row, stop_col-start_col)


## Data #######################
# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# transform data
training_size = 1000
# train data
images = x_train[:training_size].reshape((training_size, 784)) / 255
labels = np.array([embedding(y) for y in y_train[:training_size]]).reshape(
    (training_size, -1))

# test data
test_images = x_test.reshape((len(x_test), 784)) / 255
test_labels = np.array([embedding(y)
                       for y in y_test]).reshape((len(y_test), -1))

## Controls ###################
epochs = 300
alpha = 2
batch_size = 128
num_batches = int(len(images)/batch_size)

# Architecture
input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16
num_labels = 10
hidden_layer_size = ((input_rows - kernel_rows) *
                     (input_cols - kernel_cols)) * num_kernels

## Weights ###################
kernels = 0.02*np.random.random((kernel_rows*kernel_cols, num_kernels))-0.01
wt_1_2 = 0.2*np.random.random((hidden_layer_size, num_labels))-0.1

# Visualization
plot_freq = 10
train_accuracy_plot = np.zeros(int(epochs/plot_freq))

## Training ###################
for epoch in range(epochs):

    correct_cnt = 0

    # Mini Batch
    for batch in range(num_batches):

        batch_start = batch * batch_size
        batch_stop = batch_start + batch_size

        l0_raw = images[batch_start:batch_stop].reshape(-1, 28, 28)

        # Slice up each image into a bunch of kernel size images and append all images to a list
        # Each image will be sliced into 625 kernel size images
        # This is our "convolution"
        sects = list()
        for row_start in range(input_rows - kernel_rows):
            for col_start in range(input_cols - kernel_cols):
                sect = get_image_section(l0_raw,
                                         row_start,
                                         row_start+kernel_rows,
                                         col_start,
                                         col_start+kernel_cols)
                # assert sect.shape == ( 100, 1, 3, 3), f"Expected (100,1,3,3) actual {sect.shape}"
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)

        # Now we have each image in the batch, cut up into many small images
        expanded_shape = expanded_input.shape
        # 100 images cut up into 625 3*3 mini-images
        # assert expanded_shape == (100, 625, 3, 3), f"Expected (100,625,3,3) actual {expanded_shape}"

        # Now lets take all those images and stack them in an array
        flattened_input = expanded_input.reshape(
            expanded_shape[0]*expanded_shape[1], -1)
        # assert (flattened_input.shape == (62500, 9)), f"Expected (62500,9) actual {flattened_input.shape} "

        # Forward prop the transformed input through our kernels (aka weights)
        kernel_output = flattened_input.dot(kernels)
        # assert (kernel_output.shape == (62500, 16)), f"Expected (62500,16) actual {kernel_output.shape}"

        # Now reassemble the mini-images into single images
        l1 = kernel_output.reshape(expanded_shape[0], -1)

        # Apply layer 1 activation
        l1 = tanh(l1)

        # dropout
        # The dropout mask is 0s and 1s
        dropout_mask = np.random.randint(2, size=l1.shape)
        # Apply dropout mask and double the value at each element to maintain the signal after reduction by dropout
        l1 *= dropout_mask * 2

        # Predict
        l2 = softmax(l1.dot(wt_1_2))
        # assert (l2.shape == (100, 10)), f"Expect (100,10), actual {l2.shape}"

        # Compare
        # ERROR CALC TBD
        # Accuracy
        for pred, label in zip(l2, labels[batch_start:batch_stop]):
            correct_cnt += int(np.argmax(pred) == np.argmax(label))

        # Adjust
        delta = (l2 - labels[batch_start:batch_stop])

        # Delta back prop
        l2_delta = delta / (batch_size * l2.shape[0])
        l1_delta = l2_delta.dot(wt_1_2.T) * tanh_deriv(l1)
        l1_delta *= dropout_mask
        l1_delta_reshape = l1_delta.reshape(kernel_output.shape)

        # Adjust Weights
        wt_1_2 -= l1.T.dot(l2_delta) * alpha

        kernels -= flattened_input.T.dot(l1_delta_reshape) * alpha

## Testing ###################

    test_correct_cnt = 0
    
    # Predict
    

    

    # Forward Prop with convolution

    # Compare
    # Accuracy

    ## Visualize ###################
    if epoch % plot_freq == 0:

        train_correct = correct_cnt / float(len(images))
        # console
        print(epoch, "TRAIN ACC -> ", train_correct)
        # accuracy plots
        train_accuracy_plot[int(epoch / plot_freq)] = train_correct
        # weight color plots?

# Graph Error
fig, ax = plt.subplots() # an empty figure with no Axes
ax.plot(range(len(train_accuracy_plot)),train_accuracy_plot, label=('train'))
# ax.plot(range(len(test_accuracy_plot)), test_accuracy_plot, label=('test'))
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.set_title("mnist cnn")
ax.legend()
plt.show()