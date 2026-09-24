import math


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        # self.threshold = threshold
        self.bias = bias
        self.last_inputs = []
        self.last_total_z = 0
        self.last_pred = 0
        self.learning_rate = 0

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, inputs):
        total = 0
        self.last_inputs = inputs
        for x, w in zip(inputs, self.weights):
            total += x * w

        total = total + self.bias
        self.last_total_z = total

        # if total >= self.threshold:
        # return 1
        # else:
        #   return 0
        self.last_pred = self.sigmoid(total)
        return self.last_pred

    def backward_output(self, target, learning_rate):

        dL_dy = 2 * (self.last_pred - target)
        dy_dz = self.last_pred * (1 - self.last_pred)
        dz_dw = self.last_inputs
        dL_dz = dL_dy * dy_dz
        dz_db = 1
        dL_db = dL_dz * dz_db
        dL_dx = []

        self.learning_rate = learning_rate
        grads = []
        for x in self.last_inputs:
            grads.append(x * dL_dz)

        for w in self.weights:
            dL_dx.append(w * dL_dz)

        for i, (g, w) in enumerate(zip(grads, self.weights)):
            self.weights[i] = w - (g * learning_rate)
        self.bias = self.bias - (dL_db * learning_rate)
        return dL_dx

    def backward(self, incoming_grad, learning_rate):

        # incoming gradient is dL/dy
        dy_dz = self.last_pred * (1 - self.last_pred)
        dL_dz = incoming_grad * dy_dz
        dz_db = 1
        dL_db = dL_dz

        dL_dx = []
        grads = []

        for x in self.last_inputs:
            grads.append(x * dL_dz)

        for w in self.weights:
            dL_dx.append(w * dL_dz)

        for i, grad in enumerate(grads):
            self.weights[i] -= learning_rate * grad

        self.bias -= dL_dz * learning_rate
        return dL_dx


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.predict(inputs))
        return outputs

    def backward(self, incoming_grad, learning_rate):
        grads = []  # dL/dx
        collapsed = []
        for neuron, grad in zip(self.neurons, incoming_grad):
            grads.append(neuron.backward(grad, learning_rate))

        for i in range(len(grads[0])):
            total = 0
            for row in grads:
                total += row[i]

            collapsed.append(total)

        return collapsed


class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, target, learning_rate):

        output_layer = self.layers[-1]
        incoming_grad = []
        for neuron in output_layer.neurons:
            dL_dy = 2 * (neuron.last_pred - target)
            incoming_grad.append(dL_dy)

        grads = output_layer.backward(incoming_grad, learning_rate)

        for i in range((len(self.layers) - 2), -1, -1):
            grads = self.layers[i].backward(grads, learning_rate)


def loss(target, prediction):
    loss = (target - prediction) ** 2
    return loss
