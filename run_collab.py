#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 19:50:12 2017

@author: charley
"""

from __future__ import division

import logging
import math
import copy
import os

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from libact.models import SVM

from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.base.interfaces import Model
from libact.query_strategies import UncertaintySampling, RandomSampling, QUIRE, HintSVM
import libact.models
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.labelers import IdealLabeler

from CollaborativeStrategy import CollaborativeStrategy

def split_train_test(n_classes, n_labeled):
    from sklearn.datasets import load_digits

    digits = load_digits(n_class=n_classes)  # consider binary case
    X = digits.data
    y = digits.target
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    while not len(np.unique(y_train[:n_labeled])) == n_classes :
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)
   
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labled = Dataset(X_train, y_train)

    return trn_ds, tst_ds, fully_labled 

def split_train_test_from_path(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)

    while len(np.unique((y_train[:n_labeled]))) != 2:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size)

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds

if __name__=="__main__":
    #diabetes dataset
    #ds_name = 'diabetes'
    #heart dataset
    #ds_name = 'heart'
    #ds_name = 'australian'
    #dataset_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.txt' % ds_name)
    test_size = 0.33
    
    n_labeled = 5
    n_classes = 2
    
    #digit dataset
    trainSet, testSet, fully_labeled = split_train_test(n_classes, n_labeled)
    print (trainSet)
    b = SVM(kernel='linear', decision_function_shape='ovr')
    
    
    #list of query strategies
    qs = UncertaintySampling(trainSet, model=SVM(decision_function_shape='ovr'))
    randomSampling = RandomSampling(trainSet)
    
    quire = QUIRE(trainSet)
    listQueryStrategies = [qs, randomSampling, quire]
    
    collab = CollaborativeStrategy(trainSet, baseModel = b, queryStrategies = listQueryStrategies)



