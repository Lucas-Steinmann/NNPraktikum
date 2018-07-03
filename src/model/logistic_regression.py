# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from util.loss_functions import AbsoluteError
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weights : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        self.errorFunction = AbsoluteError()

        # Initialize the weight vector with small values
        self.weights = 0.01 * np.random.randn(self.trainingSet.input.shape[1])
        self.threshold = np.random.rand(1)[0]

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            for example, label in zip(self.trainingSet.input, self.trainingSet.label):
                activation, derivative = self.fire(example)
                loss = self.loss(label, activation)
                lossgrad = (-1 if label == 1 else 1) * loss
                outgrad = lossgrad * derivative
                #if verbose:
                #    print(f"Label {label}, predicted: {activation}, loss: {loss}, grad: {outgrad}")
                self.updateWeights(outgrad, example)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        activation, _ = self.fire(testInstance)
        return activation > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad, input):
        """
        Takes the gradient of the output and the forward input and updates the weights
        by the gradient descent method
        :param grad: the gradient of the output (scalar)
        :param input: the input (ndarray of input size == weight size)
        :return: None
        """
        self.weights -= self.learningRate * grad * input
        self.threshold -= self.learningRate * grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        dot = np.dot(np.array(input), self.weights)
        biaseddot = dot + self.threshold
        return Activation.sigmoid(biaseddot), Activation.sigmoidPrime(biaseddot)

    def loss(self, label, output):
        return self.errorFunction.calculateError(label, output)
