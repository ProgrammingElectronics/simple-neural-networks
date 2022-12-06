from vector_functions import *

ih_wgt = [
    [0.1, 0.2, -0.1],  # hid[0]
    [-0.1, 0.1, 0.9],  # hid[1]
    [0.1, 0.4, 0.1]]  # hid[2]


# hid[0] hid[1] hid[2]
hp_wgt = [
    [0.3, 1.1, -0.3],  # hurt?
    [0.1, 0.2, 0.0],  # win?
    [0.0, 1.3, 0.1]]  # sad?

weights = [ih_wgt, hp_wgt]

# data
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]  # Win-loss record
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]


def neural_network(input, weights):

    hidden = [0,0,0]
    output = [0,0,0]

    for i in range(len(input)):
        hidden[i] = dot_product(input, weights[0][i])
    
    for i in range(len(hidden)):
        output[i] = dot_product(hidden, weights[1][i])

    return output

pred = neural_network(input, weights)

print(pred)