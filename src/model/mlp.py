import numpy as np

from util.loss_functions import *
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def initialize_plot():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    red = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss', color=red)
    ax1.tick_params(axis='y', labelcolor=red)

    blue = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=blue)
    ax2.tick_params(axis='y', labelcolor=blue)
    return ax1, ax2


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 num_classes=10, loss='crossentropy', learning_rate=0.01, epochs=50):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : ndarray
        valid : ndarray
        test : ndarray
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : list
        validationSet : list
        testSet : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.outputTask = output_task  # Either classification or regression
        self.output_activation = output_activation

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        if loss == 'bce':
            if num_classes != 2:
                raise ValueError("Can't use binary cross entropy for "
                                 "non-binary classification task.")
            self.loss = BinaryCrossEntropyError()
        if loss == 'crossentropy':
            self.loss = CrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + loss)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.num_classes = num_classes

        # Build up the network from specific layers
        self.layers = OrderedDict()
        self.blobs = OrderedDict()
        self.derivative = OrderedDict()

        # Input layer
        input_activation = "sigmoid"
        self.layers["input"] = LogisticLayer(train.input.shape[1], 128,
                                         None, input_activation, False)
        self.layers["class_scores"] = LogisticLayer(128, self.num_classes,
                                             None, output_activation, True)

        # Output layer
        #self.layers["class_scores"] = LogisticLayer(127, self.num_classes, None, output_activation, True)
        self.inputWeights = input_weights

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        :param inp : ndarray
            a numpy array containing the input of the layer
        """
        blob = inp
        self.blobs["data"] = blob
        for name, layer in self.layers.items():
            blob = layer.forward(blob)
            self.blobs[name] = blob
        return blob

    def _compute_error(self, target):
        """
        Compute the total error of the network (error terms from the output layer)

        :return ndarray :
            a numpy array (1,1) containing the loss
        """
        error = self.loss.calculateError(target, self.blobs["class_scores"])
        return error

    def _update_weights(self, top_gradient, learning_rate):
        """ Update the weights of the layers by propagating back the error """
        gradient = top_gradient
        for name, layer in reversed(self.layers.items()):
            gradient = layer.computeDerivative(gradient, self.blobs[name])
            layer.updateWeights(learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        :param verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        epoch_losses = []

        for epoch in range(self.epochs):
            ax1, ax2 = initialize_plot()
            epoch_start = time.time()
            epoch_loss = 0.0

            if epoch == 40:
                self.learning_rate /= 5

            for example, label in zip(self.trainingSet.input, self.trainingSet.label):
                # Forward
                self._feed_forward(example)

                # Compute loss for montoring
                loss = self._compute_error(label)
                epoch_loss += loss

                # Backward
                loss_grad = self.loss.calculateGradient(label, self.blobs["class_scores"])
                self._update_weights(loss_grad, self.learning_rate)

            if verbose:
                print("Epoch %d took %d ms and had a total loss of %f"
                      % (epoch, int((time.time()-epoch_start)*1000), epoch_loss))

                epoch_losses.append(epoch_loss)
                ax1.plot(list(range(epoch+1)), epoch_losses, color='tab:red')

                self.performances.append(self.validation_score())
                ax2.plot(list(range(epoch+1)), self.performances, color='tab:blue')

                plt.show()

    def validation_score(self):
        val_pred = self.evaluate(test=self.validationSet)
        val_labels = np.argmax(self.validationSet.label, axis=1)
        return accuracy_score(val_labels, val_pred)

    def classify(self, test_instance):
        self._feed_forward(test_instance)
        return np.argmax(self.blobs["class_scores"])


    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        :param test : the dataset to be classified
            if no test data, the test set associated to the classifier will be used
        :return list:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                             axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
