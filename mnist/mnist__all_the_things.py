import numpy as np
import matplotlib.pyplot as plot
from keras.datasets import mnist

# Random
np.random.seed(1)

# Activation Functions
def relu(x):
    return (x >= 0) * x

def relu_deriv(x):
    return (x >= 0)

# Create Embeddings
def create_embedding(val):
    embedding = np.zeros((1,10))
    embedding[0,val] = 1
    return embedding

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Transformation
training_size = 1000

## Training Data
labels = np.array([create_embedding(y) for y in y_train[:training_size]]).reshape((training_size,10))
images = x_train[:training_size].reshape(training_size,784) / 255

assert labels.shape == (training_size, 10) , f"Expect (1000, 10), actual {labels.shape}"
assert images.shape == (training_size, 784) , f"Expect (1000, 784, actual {images.shape}"

## Test Data
test_labels = np.array([create_embedding(y) for y in y_test]).reshape((len(y_test),10))
test_images = x_test.reshape(len(x_test),784) / 255

assert test_labels.shape == (len(y_test), 10), f"Expect (10000, 10), actual {test_labels.shape}"
assert test_images.shape == (len(x_test), 784) , f"Expect (1000, 784, actual {test_images.shape}"

# Controls
alpha = 0.1
epochs = 300
mini_batch_size = 100
print_freq = 10
graph_array_size = int((epochs/print_freq)) + 1

# Network Architecture
hidden_size = 100

# Weights
w_0_1 = 0.2 * np.random.random((784, hidden_size)) - 0.1
w_1_2 = 0.2 * np.random.random((hidden_size, 10)) - 0.1

train_err_array = np.zeros((graph_array_size))
train_acc_array = np.zeros((graph_array_size))
test_err_array = np.zeros((graph_array_size))
test_acc_array = np.zeros((graph_array_size))

## Begin training and testing ##################################
for epoch in range(epochs + 1):

## Training ##################################
    # Metrics
    train_error = 0.0
    train_correct_count = 0

    # Mini-batch
    for i in range(int(len(labels)/mini_batch_size)):
        
        # Mini Batch
        images_mini_batch = images[i * mini_batch_size : i * mini_batch_size + mini_batch_size]
        labels_mini_batch = labels[i * mini_batch_size : i * mini_batch_size + mini_batch_size]

        # Drop Out Mask
        drop_out_mask = np.random.randint(2, size=(mini_batch_size,hidden_size))

        # Predict #
        l0 = images_mini_batch
        l1 = relu(l0.dot(w_0_1)) * drop_out_mask * 2 # 1/2 of nodes are dropped out, so increase signal by 2 during training
        l2 = l1.dot(w_1_2)
        
        assert l0.shape == (mini_batch_size,784) , f"Expect ({mini_batch_size}, 784), actual {l0.shape}" 
        assert l1.shape == (100,hidden_size) , f"Expect (100,{hidden_size}), actual {l1.shape}"
        assert l2.shape == (100,10) , f"Expect (100,10), actual {l2.shape}"

        # Compare #
        # Error and Accuracy
        train_error += np.sum((l2 - labels_mini_batch) ** 2)
        
        for i in range(mini_batch_size):
            train_correct_count += np.argmax(labels_mini_batch[i]) == np.argmax(l2[i])
            
        # Update Weights
        l2_delta = (l2 - labels_mini_batch) / mini_batch_size # prediction - actual
        l1_delta = l2_delta.dot(w_1_2.T)*relu_deriv(l1) * drop_out_mask 
        
        l2_wt_delta = l2_delta.T.dot(l1)
        l1_wt_delta = l1_delta.T.dot(l0) 

        w_1_2 -= l2_wt_delta.T * alpha
        w_0_1 -= l1_wt_delta.T * alpha

## Testing ##################################
    
    # Metrics
    test_error = 0
    test_correct_count = 0

    for test_label, test_image in zip(test_labels, test_images):
    
        # Predict
        l0 = test_image
        l1 = relu(l0.dot(w_0_1))
        l2 = l1.dot(w_1_2)

        # Compare
        test_error += np.sum((l2 - test_label)**2)
        test_correct_count += np.argmax(l2) == np.argmax(test_label)

    # Printing and Graphing Results
    train_err_percent = train_error / float(len(labels))
    train_acc_percent = train_correct_count / float(len(labels))
    test_err_percent = test_error / float(len(test_labels))
    test_acc_percent = test_correct_count / float(len(test_labels))

    if(epoch % print_freq == 0):
        # Print
        print(epoch, "Train_Err", "{:.3f}".format(train_err_percent), \
                     "Train_Acc", "{:.3f}".format(train_correct_count / float(len(labels))), \
                     "Test_Err", "{:.3f}".format(test_error / float(len(test_labels))), \
                     "Test_Acc", "{:.3f}".format(test_correct_count / float(len(test_labels))))

        # Graph
        index = int(print_freq/print_freq)
        train_err_array[index] = train_err_percent
        train_acc_array[index] = train_acc_percent
        test_err_array[index] = test_err_percent
        test_acc_array[index] = test_acc_percent
    
# Graphing Results
