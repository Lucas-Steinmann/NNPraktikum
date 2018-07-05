from abc import ABCMeta, abstractmethod


class Layer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, input):
        """
        Compute forward step over the input using its weights

        :param input: ndarray
            a numpy array (1,nIn + 1) containing the input of the layer
        :return ndarray: a numpy array (1,nOut) containing the output of the layer
        """
        pass

    @abstractmethod
    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        :param nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        :param nextWeights: ndarray
            a numpy array containing the weights from next layer
        :return ndarray: a numpy array containing the partial derivatives on this layer
        """
        pass

    @abstractmethod
    def updateWeights(self, delta, learning_rate):
        """ Update the weights of the layer """
        pass
