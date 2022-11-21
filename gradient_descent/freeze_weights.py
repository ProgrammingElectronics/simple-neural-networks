def neural_network(inputs, weights, alpha):

    assert(len(inputs) == len(weights))

    temp = [0,0,0]

    # Predict

    ## elementwise input * weight
    for element in range(len(inputs)):
        temp[element] = inputs[element] * weights[element]

    ## sum for dot product
    pred = sum(temp)

    # print(temp)
    # print(pred)
   
    # Compare
    mse = (pred - actual) ** 2
    print("error: "+ str(mse))
   
    # Adjust
    delta = pred - actual

    weight_deltas = [0,0,0]
        
    ## Adjust each weight by delta scaled by input
    for element in range(len(weights)):

        if(element != 0):
            weight_deltas[element] = delta * input[element]
            weights[element] -= weight_deltas[element] * alpha 

    print("weights: " + str(weights))

# Weights
weights = [0.1, 0.2, -.1]

# Input Data
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]
input = [toes[0],wlrec[0],nfans[0]]

# Actual
win_or_lose_binary = [1, 1, 0, 1]
actual = win_or_lose_binary[0]

# Knobs
alpha = 0.3
epochs = 20


for epoch in range(epochs):
    neural_network(input, weights, alpha)