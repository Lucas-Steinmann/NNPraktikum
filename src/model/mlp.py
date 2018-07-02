from datetime import datetime

import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier

from sklearn.metrics import accuracy_score

class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='ce', learningRate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : DataSet
        valid : DataSet
        test : DataSet
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : DataSet
        validationSet : DataSet
        testSet : DataSet
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'ce':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + str)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers

        # Build up the network from specific layers
        self.layers = []

        # Input layer
        inputActivation = "relu"
        self.layers.append(LogisticLayer(train.input.shape[1], 512,
                           None, inputActivation, False, len(train.input)))

        # Hidden layer
        hiddenActivation = "relu"
        self.layers.append(LogisticLayer(512, 512,
                           None, hiddenActivation, False, len(train.input)))

        hiddenActivation = "sigmoid"
        self.layers.append(LogisticLayer(512, 512,
                           None, hiddenActivation, False, len(train.input)))

        # Output layer
        outputActivation = "softmax"
        self.layers.append(LogisticLayer(512, 10,
                           None, outputActivation, True, len(train.input)))

        self.inputWeights = inputWeights

        # we don't add bias here since we are doing it during forward pass

        # add bias values ("1"s) at the beginning of all data sets
        # self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
        #                                     axis=1)
        # self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
        #                                       axis=1)
        # self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _to_one_hot(self, label):
        tmp = np.zeros(10)
        tmp[label] = 1
        return tmp

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        addBias = ( lambda data: np.append(data, 1.0) )

        temp = inp
        for l in self.layers:
            temp = l.forward(addBias(temp))
        return temp
        
    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        return self.loss.calculateError(self._to_one_hot(target), self._get_output_layer().outp)
    
    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for l in self.layers:
            l.updateWeights(learningRate)

        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        if verbose:
            result = self.evaluate(self.validationSet)
            accuracy = accuracy_score(self.validationSet.label, result)
            print('Before training: Accuracy: {0}'.format(accuracy))


        for i in range(self.epochs):

            # Decreasing learning rate
            if i == 3:
                self.learningRate /= 5
                print 'New learning rate:', self.learningRate
            if i == 10:
                self.learningRate /= 3
                print 'New learning rate:', self.learningRate
            if i == 20:
                self.learningRate /= 3
                print 'New learning rate:', self.learningRate

            start = datetime.now()
            for inp, label in zip(self.trainingSet.input, self.trainingSet.label):
                output = self._feed_forward(inp)
                deltas = self.loss.calculateDerivative(self._to_one_hot(label), output)

                for layer in range(-1, -len(self.layers) - 1, -1):
                    if layer == -1:
                        next_weights = 1.0
                    else:
                        next_weights = self._get_layer(layer + 1).weights[:-1]
                    deltas = self._get_layer(layer).computeDerivative(deltas, next_weights)

                self._update_weights(self.learningRate)

            if verbose:
                result = self.evaluate(self.trainingSet)
                accuracy_training = accuracy_score(self.trainingSet.label, result)

                result = self.evaluate(self.validationSet)
                accuracy_validation = accuracy_score(self.validationSet.label, result)
                self.performances.append(accuracy_validation)

                end = datetime.now()
                print('Iteration {0}: Accuracy Training: {1}, Accuracy Validation: {2} (time: {3})'.
                      format(i, accuracy_training, accuracy_validation, end - start))

                result = self.evaluate(self.trainingSet)
                accuracy = accuracy_score(self.trainingSet.label, result)


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        return np.argmax(self._feed_forward(test_instance))
        

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

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
