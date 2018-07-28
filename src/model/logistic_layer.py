import time

import numpy as np

from util.activation_functions import Activation
from model.layer import Layer


class LogisticLayer(Layer):
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    gradient : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.derivative = Activation.getDerivative(self.activationString)

        self.nIn = nIn+1
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((self.nIn, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.activations = np.ndarray((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, self.nIn))-0.5
        else:
            self.weights = weights
        self.gradient = np.zeros_like(self.weights)

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, input):
        self.input = np.insert(input, 0, 1)
        self.output = self.weights.dot(self.input)
        self.activations = self.activation(self.output)
        return self.activations

    def computeDerivative(self, nextDerivatives, nextActivation):
        da_dz = self.derivative(self.output)
        if len(da_dz.shape) == 1:
            # did not get jacobi -> make it to matrix
            da_dz = np.diag(da_dz)
        dE_do = np.dot(nextDerivatives, da_dz)
        dE_dw = np.outer(dE_do, self.input)
        self.gradient = dE_dw
        return np.dot(self.weights.T, dE_do)[1:]

    def updateWeights(self, learning_rate):
        """ Update the weights of the layer """
        self.weights -= self.gradient * learning_rate

