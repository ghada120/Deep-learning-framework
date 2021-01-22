import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.error_per_epoch = []

    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # set loss to use
    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    # train the network
    def train(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output, x_train[j])

                # backward propagation
                error = self.loss_derivative(y_train[j], output, x_train[j])  # dE/dY
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            self.error_per_epoch.append(err)
            print('error={}         for epoch {} of {}' .format(err, i + 1, epochs))

        # visualization
        plt.ion()
        fig = plt.figure()
        plt.axis([0, epochs, 0, max(self.error_per_epoch)])

        i = 0

        while i < epochs:
            plt.scatter(i, self.error_per_epoch[i])
            i += 1
            plt.show()
            plt.pause(0.000001)

    # predict output for given input
    def predict_output(self, input_data):
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result



