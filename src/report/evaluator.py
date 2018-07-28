# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np


class Evaluator:
    """
    Print performance of a classification model over a dataset
    """

    def printTestLabel(self, testSet):
        # print all test labels
        for label in testSet.label:
            print(label)

    def printResultedLabel(self, pred):
        # print all test labels
        for result in pred:
            print(result)

    def printComparison(self, testSet, pred):
        for label, result in zip(testSet.label, pred):
            print("Label: %r. Prediction: %r" % (bool(label), bool(result)))

    def printClassificationResult(self, testSet, pred, targetNames):
        labels = np.argmax(testSet.label, axis=1)
        print(classification_report(labels,
                                    pred,
                                    target_names=targetNames))

    def printConfusionMatrix(self, testSet, pred):
        print(confusion_matrix(testSet.label, pred))

    def printAccuracy(self, testSet, pred):
        print("Accuracy of the recognizer: %.2f%%" %
              (average_precision_score(testSet.label, pred)*100))
