# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np
from numpy import exp
from numpy import divide
from numpy import tanh
from numpy import square


class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        if np.isscalar(netOutput):
            return -1 if netOutput < 0 else (0 if netOutput == 0 else 1)
        signed = np.ndarray(netOutput.shape)
        signed[netOutput > threshold] = 1
        signed[netOutput == threshold] = 0
        signed[netOutput < threshold] = -1
        return signed

    @staticmethod
    def sigmoid(netOutput):
        # Here you have to code the sigmoid function
        return 1.0/(1.0+exp(-netOutput))

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        return Activation.sigmoid(netOutput)*(1-Activation.sigmoid(netOutput))

    @staticmethod
    def tanh(netOutput):
        # Here you have to code the tanh function
        return tanh(netOutput)

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        return 1-square(tanh(netOutput))

    @staticmethod
    def rectified(netOutput):
        rectified = np.copy(netOutput)
        rectified[netOutput < 0] = 0
        return rectified

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        rectified = np.copy(netOutput)
        rectified[netOutput < 0] = 0
        rectified[netOutput >= 0] = 1
        return rectified

    @staticmethod
    def identity(netOutput):
        return netOutput

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        return 1.0

    @staticmethod
    def softmax(netOutput):
        exponated = exp(netOutput)
        return exponated/np.sum(exponated)

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
