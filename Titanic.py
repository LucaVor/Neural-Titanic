import random
import math

trainingCSV = open("train.csv", "r")
trainingRawText = trainingCSV.read()
trainingLines = trainingRawText.split("\n")[1:]

stringInputs = []
desiredOutputs = []

inputs = []
inputBatches = []

maxInputs = []

averageAge = 0

#TODO: NORMALIZE ALL INPUTS TO  01

for line in trainingLines:
    dataPoints = line.split(",")

    if len(dataPoints) == 1:
        break

    while len(maxInputs) < 9:
        maxInputs.append(0)

    maxInputs[0] = max(maxInputs[0], float(dataPoints[0]))
    maxInputs[1] = max(maxInputs[1], float(dataPoints[2]))
    maxInputs[2] = 1
    if (dataPoints[6].isnumeric()):
        maxInputs[3] = max(maxInputs[2], float(dataPoints[6]))

    maxInputs[4] = max(maxInputs[4], float(dataPoints[7]))
    maxInputs[5] = max(maxInputs[5], float(dataPoints[8]))
    maxInputs[6] = max(maxInputs[6], float(dataPoints[10]))

    maxInputs[7] = 3
    maxInputs[8] = 1

    stringInputs.append(dataPoints)
    desiredOutputs.append(int(dataPoints[1]) * 2 - 1)

def returnInputs(ind):
    try:
        x = float(stringInputs[ind][6])
    except:
        stringInputs[ind][6] = averageAge

        # PID    PCLASS  SEX    AGE    SIBSP   PARCH    FARE   EMBARKED  Married
        # 0.11   -1.58  -1.24  -0.4   -0.5    -0.09    -0.06   0.61       

    return [float(stringInputs[ind][0]) / maxInputs[0], #Passenger ID
            float(stringInputs[ind][2]) / maxInputs[1], #Pclass
            (1 if stringInputs[ind][5] == "male" else 0) / maxInputs[2], #Sex
            float(stringInputs[ind][6]) / maxInputs[3], #Age
            float(stringInputs[ind][7]) / maxInputs[4], #SibSp
            float(stringInputs[ind][8]) / maxInputs[5], #Parch
            float(stringInputs[ind][10]) / maxInputs[6],#Fare
            (3 if stringInputs[ind][12] == "Q" else (2 if stringInputs[ind][12] == "C" else 1)) / maxInputs[7],
            (1 if "Mrs." in stringInputs[ind][4] else 0) / maxInputs[8]]

for i in range(len(stringInputs)):
    inputs.append(returnInputs(i))

currentBatch = []
batchSize = 891

for i in range(len(inputs)):
    currentBatch.append(inputs[i])

    if (i + 1) % batchSize == 0:
        inputBatches.append(currentBatch)
        currentBatch = []        

def sigmoid(x):
    return math.tanh(x)

def weighted_sum(values, weights, bias):
    total_sum = 0
    for i in range(len(weights)):
        total_sum += values[i] * weights[i]
    return sigmoid(total_sum + bias)

weights = []

bias = 0
num_of_inputs = len(returnInputs(1))

for i in range(num_of_inputs):
    weights.append(random.random() * 2 - 1)

def total_cost(samples):
    cost = 0

    for i in range(0, len(samples)):
        current_input = samples[i]
        output = weighted_sum(current_input, weights, bias)

        cost += (((desiredOutputs[i] - output) * (desiredOutputs[i] - output)))

    return cost

def count_correct(samples):
    c = 0
    for i in range(0, len(samples)):
        current_input = samples[i]
        output = weighted_sum(current_input, weights, bias)
        thinks = output > 0
        real = desiredOutputs[i] > 0

        if thinks == real:
            c = c + 1

    return c

epoch_count = 1200

learning_rate = 1.0e-4
h = 0.000001

print(f"Starting: {count_correct(inputs)/891}: {total_cost(inputs)}")

gradient = []
for i in range(0, len(weights)):
    gradient.append(0)

for epoch in range(epoch_count):
    for batch in inputBatches:
        original_cost = total_cost(batch)

        for i in range(0, len(weights)):
            weights[i] = weights[i] + h
            new_cost = total_cost(batch)
            weights[i] = weights[i] - h 

            gradient[i] = (new_cost - original_cost) / h

        bias = bias + h
        new_cost = total_cost(batch)
        bias = bias - h

        bias = bias - ((new_cost - original_cost) / h) * learning_rate

        for i in range(0, len(gradient)):
            weights[i] = weights[i] - gradient[i] * learning_rate

    if epoch % 100 == 0:
        print(f"{epoch}: {count_correct(inputs)}/891 or {count_correct(inputs)/891}% Cost: {total_cost(inputs)}")


gotRight = 0
total = 0

for i in range(0, len(stringInputs)):
    current_input = inputs[i]
    output = weighted_sum(current_input, weights, bias)
    guess = output > 0
    real = desiredOutputs[i] > 0
    right = guess == real

    gotRight += (1 if right else 0)
    total += 1

    print(str(1 if guess else 0) + " : " + str(1 if real else 0) + " : " + str(right))

# 0 2 6 7 8 10

for i in range(len(weights)):
    print(f"Weight {i}: " + str(weights[i]))

print(weights)
print("Bias: " + str(bias))

print (f"Got {gotRight} out of {total} or {round((float(gotRight)/float(total)) * 1000) / 1000}%")