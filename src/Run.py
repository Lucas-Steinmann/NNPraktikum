#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from data.mnist import MNIST
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron
from report.evaluator import Evaluator


def main():
    dataSeven = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)
    data = MNIST("../data/mnist_seven.csv", 3000, 1000, 1000)
    #myStupidClassifier = StupidRecognizer(dataSeven.trainingSet,
    #                                      dataSeven.validationSet,
    #                                      dataSeven.testSet)
    #myPerceptronClassifier = Perceptron(dataSeven.trainingSet,
    #                                    dataSeven.validationSet,
    #                                    dataSeven.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=30)
    #myLRClassifier = LogisticRegression(dataSeven.trainingSet,
    #                                    dataSeven.validationSet,
    #                                    dataSeven.testSet,
    #                                    learningRate=0.005,
    #                                    epochs=60,
    #                                    learning_rate_step=30)
    myMLPClassifier = MultilayerPerceptron(data.trainingSet,
                                           data.validationSet,
                                           data.testSet,
                                           learning_rate=0.005,
                                           epochs=60)

    # Train the classifiers
    print("=========================")
    print("Training..")

    #print("\nStupid Classifier is training..")
    #myStupidClassifier.train()
    #print("Done..")

    #print("\nPerceptron is training..")
    #myPerceptronClassifier.train()
    #print("Done..")

    #print("\nLogistic Regression is training..")
    #myLRClassifier.train()
    #print("Done..")

    print("\Multilayer Perceptron is training..")
    myMLPClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()
    #lrPred = myLRClassifier.evaluate()
    mlpPred = myMLPClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.testSet, stupidPred)

    print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, perceptronPred)

    print("\nResult of the Logistic Regression recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, lrPred)

    print("\nResult of the MultiLayer Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.testSet, mlpPred)


if __name__ == '__main__':
    main()
