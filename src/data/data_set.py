# -*- coding: utf-8 -*-

import numpy as np


class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data: np.ndarray, targetDigit=None, num_classes=10):

        # The label of the digits is always the first fields
        # Doing normalization
        self.input = 1.0 * data[:, 1:] / 255
        if targetDigit:
            self.label = data[:, 0]
            self.label = list(map(lambda a:
                                  1 if str(a) == targetDigit
                                  else 0, self.label))
            self.num_classes = 2
        else:
            self.num_classes = num_classes
            self.label = np.zeros((data.shape[0], num_classes))
            self.label[[range(data.shape[0])], data[:,0]] = 1
        self.targetDigit = targetDigit

    def __iter__(self):
        return self.input.__iter__()
