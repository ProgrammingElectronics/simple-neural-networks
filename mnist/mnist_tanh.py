import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

np.random.seed(1)

## Functions ################################

# Activations
def tanh(x):

    return np.tanh(x)

def tanh_deriv(x):

    return 1 - (x ** 2)

def softmax(x):

    temp = np.exp(x)
    return temp / np.sum(temp, axis=1,keepdims=True) 

# Embeddings
def embedding(x):

    embedding = np.zeros((1,10))
    embedding[0,x] = 1
    return embedding

## Data
# Import Data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

# Transform Data
num_train_samples = 1000

images = np.reshape(x_train[:num_train_samples], (num_train_samples, 784)) / 255
labels = np.reshape([embedding(y) for y in y_train[0:num_train_samples]], (num_train_samples, 10))

test_images = np.reshape(x_test, (len(x_test), 784)) / 255
test_labels = np.reshape([embedding(y) for y in y_test], (len(y_test), 10))

## Controls
alpha = 2
epochs = 300

mini_batch_size = 100
num_mini_batches = int(len(images)/mini_batch_size)

## Architecture
hidden_layer_size = 100

## Weights
wt_0_1 = 0.02 * np.random.random((784, hidden_layer_size)) - 0.01
wt_1_2 = 0.2 * np.random.random((hidden_layer_size, 10)) - 0.1

## Reporting
freq = 10 
train_accuracy_plot = np.zeros((int(epochs/freq))) 
test_accuracy_plot = np.zeros((int(epochs/freq))) 

## Training
for epoch in range(epochs):
    
    correct_cnt = 0

    for batch_index in range(num_mini_batches):
        
        start_batch = batch_index * mini_batch_size
        stop_batch = (batch_index + 1) * mini_batch_size
        drop_out_mask = np.random.randint(2,size=(mini_batch_size, hidden_layer_size)) 

        ## Predict ######################################
        l0 = images[start_batch:stop_batch]
        l1 = tanh(l0.dot(wt_0_1)) * drop_out_mask * 2
        l2 = softmax(l1.dot(wt_1_2))
        
        ## Compare ######################################
        # Error - TBD for softmax...
        # Accuracy
        batch_labels = labels[start_batch:stop_batch]

        for i in range(mini_batch_size):
            
            correct_cnt += int(np.argmax(l2[i]) == np.argmax(batch_labels[i]))

        ## Learn ######################################
        # Deltas
        l2_delta = (l2 - batch_labels) / (mini_batch_size * l2.shape[0]) # dividing by l2.shape[0] is part of derivative of softmax
        l1_delta = l2_delta.dot(wt_1_2.T) * tanh_deriv(l1) * drop_out_mask

        # Adjust Weight
        wt_1_2 -= l1.T.dot(l2_delta) * alpha
        wt_0_1 -= l0.T.dot(l1_delta) * alpha
    
    ## Testing ##
    test_correct_cnt = 0

    for test_image, test_label in zip(test_images, test_labels):

        drop_out_mask = np.random.randint(2, size=(1,hidden_layer_size))

        ## Predict
        l0 = test_image
        l1 = tanh(l0.dot(wt_0_1)) * drop_out_mask * 2
        l2 = softmax(l1.dot(wt_1_2))
        
        ## Compare
        # Accuracy
        test_correct_cnt += int(np.argmax(l2) == np.argmax(test_label))

    ## Visualize
    # Print Error
    if(epoch % freq == 0):
        train_accuracy_plot[int(epoch/freq)] = correct_cnt / float(len(labels))
        print(epoch, "Train Acc: ", train_accuracy_plot[int(epoch/freq)])
        test_accuracy_plot[int(epoch/freq)] = test_correct_cnt / float(len(test_labels))
        print(epoch, "Test Acc: ", test_accuracy_plot[int(epoch/freq)])
        
        

# Graph Error
fig, ax = plt.subplots() # an empty figure with no Axes
ax.plot(range(len(train_accuracy_plot)), train_accuracy_plot, label=('train'))
ax.plot(range(len(test_accuracy_plot)), test_accuracy_plot, label=('test'))
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.set_title("mnist all the things")
ax.legend()
plt.show()



