import numpy as np


# Define the sigmoid activation function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the sigmoid activation function.
def sigmoid_derivative(x):
    return x * (1 - x)


# Define the fully connected neural network class.
class NeuralNetwork:

    # Initialize the neural network with the given architecture.
    # The architecture is a list of integers, where each integer
    # represents the number of neurons in a layer. For example,
    # an architecture of [2, 3, 1] would create a neural network
    # with two input neurons, one hidden layer of three neurons,
    # and one output neuron.
    def __init__(self, architecture):
        # Initialize the weights and biases with random values.
        self.weights = [np.random.randn(n_inputs, n_outputs)
                        for n_inputs, n_outputs in zip(architecture[:-1], architecture[1:])]
        self.biases = [np.random.randn(n_outputs) for n_outputs in architecture[1:]]

    # Feed the given input through the neural network and
    # return the output.
    def feed_forward(self, inputs):
        # Feed the input through each layer of the network.
        for weights, biases in zip(self.weights, self.biases):
            inputs = sigmoid(np.dot(inputs, weights) + biases)
        return inputs

    # Train the neural network using the given inputs and expected outputs.
    # The learning rate and number of epochs control the training process.
    def train(self, inputs, expected_outputs, learning_rate, num_epochs):
        # Train the network for the given number of epochs.
        for epoch in range(num_epochs):
            # Feed the inputs through the network and calculate the output.
            outputs = self.feed_forward(inputs)

            # Calculate the error between the expected and actual outputs.
            error = expected_outputs - outputs

            # Backpropagate the error and update the weights and biases.
            for i in reversed(range(len(self.weights))):
                weights = self.weights[i]

                # Calculate the gradient of the error with respect to the weights.
                gradient = np.dot(inputs.T, error * sigmoid_derivative(outputs))

                # Update the weights and biases.
                self.weights[i] += learning_rate * gradient
                self.biases[i] += learning_rate * error

            # Print the current error at the end of each epoch.
            print(f'Epoch {epoch + 1}: error = {np.mean(np.abs(error))}')


# Create an instance of the NeuralNetwork class with the desired architecture.
nn = NeuralNetwork([2, 3, 1])

# Define the input and expected output for the training data.
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([[0], [1], [1], [0]])

# Train the neural network with the given inputs and expected outputs.
nn.train(inputs, expected_outputs, learning_rate=0.5, num_epochs=10000)

# Use the trained neural network to make predictions.
print(nn.feed_forward([0, 0]))  # should output a value close to 0
print(nn.feed_forward([0, 1]))  # should output a value close to 1
print(nn.feed_forward([1, 0]))  # should output a value close to 1
print(nn.feed_forward([1, 1]))  # should output a value close to 0
