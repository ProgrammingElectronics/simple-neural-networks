weight = 0.5
actual = 0.8
input = 0.5

for iteration in range(20):
    pred = input * weight
    error = (pred - actual) ** 2
    direction_and_amount = (pred - actual) * input
    weight = weight - direction_and_amount

    print("Error: " + str(error) + " Prediction: " + str(pred))