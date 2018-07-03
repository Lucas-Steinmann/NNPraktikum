# -*- coding: utf-8 -*-

import sys
import logging

from math import pow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from util.activation_functions import Activation
from util.loss_functions import AbsoluteError
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm
    """
    def __init__(self, train, valid, test, learningRate=0.01, epochs=50, learning_rate_step=0, step_factor=0.5):

        self.baseLearningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.errorFunction = AbsoluteError()
        self.learningRateStep = learning_rate_step
        self.stepFactor = step_factor

        # Initialize the weight vector with small values
        self.weights = 0.01 * np.random.randn(self.trainingSet.input.shape[1])
        self.threshold = np.random.rand(1)[0]

    def learning_rate(self, epoch):
        if self.learningRateStep != 0:
            return self.baseLearningRate * pow(self.stepFactor, 1 + (epoch//self.learningRateStep))
        return self.baseLearningRate

    def train(self, verbose=True, graph=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        fig, train_ax = plt.subplots()
        val_ax = train_ax.twinx()
        train_losses = []
        validation_scores = []
        xs = []
        for epoch in range(self.epochs):
            xs.append(epoch)
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            validation_scores.append(self.validation_score())
        train_ax.plot(xs, train_losses, color="tab:blue")
        val_ax.plot(xs, validation_scores, color="tab:red")
        fig.tight_layout()
        plt.show()

    def validation_score(self):
        predictions = self.evaluate(self.validationSet.input)
        return accuracy_score(self.validationSet.label, predictions)

    def train_epoch(self, epoch):
        """ Train the network for one epoch and return the accumulated loss """
        train_loss = 0.0
        print(f"Learning rate at epoch {epoch}: {self.learning_rate(epoch)}")
        for example, label in zip(self.trainingSet.input, self.trainingSet.label):
            # Forward
            # (and calculate derivative in one swipe,
            # otherwise the dot product would have to be cached or calculated twice)
            activation, derivative = self.fire(example)
            # Calculate loss
            loss = self.loss(label, activation)
            train_loss += loss
            # Gradient of the loss function (absolute error)
            lossgrad = (-1 if label == 1 else 1) * loss
            # Gradient of non-linearity (e.g. sigmoid)
            outgrad = lossgrad * derivative
            # Update weight (linear with input)
            self.updateWeights(outgrad, example, self.learning_rate(epoch))
        return train_loss

    def classify(self, testInstance):
        """Classify a single instance.
        :param testInstance : list of floats
        :return eturns True if the testInstance is recognized as a 7, False otherwise.
        """
        activation, _ = self.fire(testInstance)
        return activation > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.
        :param test: the dataset to be classified
        if no test data, the test set associated to the classifier will be used
        :return: List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad, input, learning_rate):
        """
        Takes the gradient of the output and the forward input and updates the weights
        by the gradient descent method
        :param grad: the gradient of the output (scalar)
        :param input: the input (ndarray of input size == weight size)
        :return: None
        """
        self.weights -= learning_rate * grad * input
        self.threshold -= learning_rate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        dot = np.dot(np.array(input), self.weights)
        biaseddot = dot + self.threshold
        return Activation.sigmoid(biaseddot), Activation.sigmoidPrime(biaseddot)

    def loss(self, label, output):
        return self.errorFunction.calculateError(label, output)
