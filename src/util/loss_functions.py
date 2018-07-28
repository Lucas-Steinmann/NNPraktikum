# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np
from math import log

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass

class DifferentiableError(Error):

    @abstractmethod
    def calculateGradient(self, target, output):
        # calculate the gradient of the error function
        pass


class AbsoluteError(DifferentiableError):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)

    def calculateGradient(self, target, output):
        # Positive if output is already greater at point otherwise negative
        gradient = np.ones_like(target)
        gradient[output < target] = -1
        return gradient


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target: np.ndarray, output: np.ndarray) -> float:
        # It is the numbers of differences between target and output
        return len((target - output).nonzero())



class MeanSquaredError(DifferentiableError):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        if target is int:
            return np.square(target - output)
        return 1.0/target.shape[0] * np.sum(np.square(target - output))

    def calculateGradient(self, target, output):
        if target is int:
            return 2 * (output - target)
        return 2.0/target.shape[0] * (output - target)


class SumSquaredError(DifferentiableError):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        return 1.0/2 * np.sum(np.square(target - output))

    def calculateGradient(self, target, output):
        return output - target


class BinaryCrossEntropyError(DifferentiableError):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        return -target*np.log(output) - (1-target)*np.log(1-output)

    def calculateGradient(self, target, output):
        return -target/output + (1-target)/(1-output)


class CrossEntropyError(DifferentiableError):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        return sum([-t*log(o) for t,o in zip(target, output)])

    def calculateGradient(self, target, output):
        return -target*(1.0/output)
