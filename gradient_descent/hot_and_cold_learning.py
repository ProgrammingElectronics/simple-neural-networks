def neural_network(input, weight):
    prediction = input * weight
    return prediction

def mean_squared_error(pred, actual):

    return (pred - actual)**2


weight = 0.5
input = 0.5
actual = 0.8
learning_rate = .001


for iteration in range(1101):

    # Make a prediction
    pred = neural_network(input, weight)

    # Compare with actual MSE
    error = mean_squared_error(pred, actual)

    if (iteration % 10) == 0:
        print(iteration, "prediction-> ",pred, "error->", error, " weight-> ", weight)

    # Learn
    # Make prediction by adjusting weight UP
    pred_up = neural_network(input, weight+learning_rate)
    error_up = mean_squared_error(pred_up, actual)
    
    # Make prediction by adjusting weight DOWN
    pred_down = neural_network(input, weight-learning_rate)
    error_down = mean_squared_error(pred_down, actual)

    # Adjust weight based on lowest error
    if error_down < error_up:
        weight -= learning_rate
    
    if error_up < error_down:
        weight += learning_rate






    

