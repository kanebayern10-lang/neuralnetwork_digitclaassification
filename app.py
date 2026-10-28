import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
xtrain = np.loadtxt('train_X.csv', delimiter=',').T
ytrain = np.loadtxt('train_label.csv', delimiter=',').T
xtest = np.loadtxt('test_X.csv', delimiter=',').T
ytest = np.loadtxt('test_label.csv', delimiter=',').T

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

# Normalize pixel values
if np.max(xtrain) > 1:
    xtrain = xtrain / 255.0
    xtest = xtest / 255.0

# Display random image
index = random.randrange(0, xtrain.shape[1])

plt.imshow(xtrain[:, index].reshape(28, 28), cmap='gray')
plt.show()


# Activation functions
def tanhh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def tanhh_derivative(x):
    return 1 - np.tanh(x)**2


def relu_derivative(x):
    return np.array(x > 0, dtype=np.float32)


# Initialize parameters
def initialize_parameters(N, H, Y):

    w1 = np.random.randn(H, N) * 0.01
    b1 = np.zeros((H, 1))

    w2 = np.random.randn(Y, H) * 0.01
    b2 = np.zeros((Y, 1))

    params = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return params


# Forward propagation
def forward_propagation(x, params):

    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    fw_cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }

    return a2, fw_cache


# Cost function
def costfunction(y, a2):

    m = y.shape[1]

    a2 = np.clip(a2, 1e-12, 1.0)

    cost = -np.sum(y * np.log(a2)) / m

    return cost


# Backward propagation
def backward_propagation(x, y, params, fw_cache):

    w2 = params["w2"]

    z1 = fw_cache["z1"]
    a1 = fw_cache["a1"]
    a2 = fw_cache["a2"]

    m = y.shape[1]

    dz2 = a2 - y

    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(
        dz2, axis=1, keepdims=True
    )

    dz1 = np.dot(w2.T, dz2) * relu_derivative(z1)

    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(
        dz1, axis=1, keepdims=True
    )

    grads = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2
    }

    return grads


# Update parameters
def update_parameters(params, grads, learning_rate):

    params["w1"] -= learning_rate * grads["dw1"]
    params["b1"] -= learning_rate * grads["db1"]

    params["w2"] -= learning_rate * grads["dw2"]
    params["b2"] -= learning_rate * grads["db2"]

    return params


# Model training
def model(x, y, H, iterations, learning_rate):

    N = x.shape[0]
    Y = y.shape[0]

    costlist = []

    params = initialize_parameters(N, H, Y)

    for i in range(iterations):

        # Forward propagation
        a2, forward_cache = forward_propagation(x, params)

        # Calculate cost
        cost = costfunction(y, a2)

        # Backward propagation
        grads = backward_propagation(
            x, y, params, forward_cache
        )

        # Update parameters
        params = update_parameters(
            params, grads, learning_rate
        )

        costlist.append(cost)

        if (i + 1) % 10 == 0:
            print(
                f"Iteration {i + 1}, Cost: {cost:.4f}"
            )

    return params, costlist


# Train the model
iterations = 100
learning_rate = 0.01
H = 1000

params, costlist = model(
    xtrain, ytrain, H, iterations, learning_rate
)


# Plot cost
plt.plot(costlist)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()


# Test the model
a2, _ = forward_propagation(xtest, params)

predictions = np.argmax(a2, axis=0)
actual = np.argmax(ytest, axis=0)

accuracy = np.mean(predictions == actual) * 100

print("Test Accuracy:", accuracy, "%")


# Display prediction
index = random.randrange(0, xtest.shape[1])

a2, _ = forward_propagation(
    xtest[:, index:index + 1], params
)

predicted_label = np.argmax(a2[:, 0])
actual_label = actual[index]

plt.imshow(
    xtest[:, index].reshape(28, 28),
    cmap='gray'
)

plt.title(
    f"Actual: {actual_label}, Predicted: {predicted_label}"
)

plt.show()