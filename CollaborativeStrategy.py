#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:44:34 2017

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

from alearner import Alearner

LOGGER = logging.getLogger(__name__)


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

def split_train_test2(dataset_filepath, test_size, n_labeled):
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


if __name__ == "__main__":
    
    #diabetes dataset
    ds_name = 'diabetes'
    #heart dataset
    #ds_name = 'heart'
    #ds_name = 'australian'
    dataset_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '%s.txt' % ds_name)
    test_size = 0.33
    
    n_labeled = 5
    n_classes = 2
    
    #digit dataset
    #trainSet, testSet, fully_labeled = split_train_test(n_classes, n_labeled)

    
    trainSet, testSet, _ ,  fully_labeled = split_train_test2(dataset_filepath,test_size, n_labeled)
    
    
    
    labeler = IdealLabeler(fully_labeled)
    
    listLearners = list()
    
    baseModel = SVM(kernel='linear', decision_function_shape='ovr')
    
    qs = UncertaintySampling(trainSet, model=SVM(decision_function_shape='ovr'))
    a1 = Alearner(qs, hisModel = SVM(kernel='linear', decision_function_shape='ovr'), hisDataset= copy.deepcopy(trainSet))
    listLearners.append(a1)
    
    randomSampling = RandomSampling(trainSet)
    a2 = Alearner(randomSampling, hisModel = SVM(kernel='linear', decision_function_shape='ovr'), hisDataset= copy.deepcopy(trainSet))
    listLearners.append(a2)
    
    quire = QUIRE(trainSet)
    a3=Alearner(quire, hisModel = SVM(kernel='linear', decision_function_shape='ovr'), hisDataset= copy.deepcopy(trainSet))
    listLearners.append(a3)
    
#    hintSv = HintSVM(copy.deepcopy(trainSet), cl=1.0, ch=1.0)
#    a4 = Alearner(hintSv, hisModel = SVM(kernel='linear', decision_function_shape='ovr'), hisDataset= copy.deepcopy(trainSet))
#    listLearners.append(a4) 

    
    E_in = []
    E_out = []
    query_num = np.arange(1, trainSet.len_unlabeled()+1)
    for j in range(trainSet.len_unlabeled()):
        baseModel.train(trainSet)
        X, _ = zip(*trainSet.data)
        votes =  {}
        for i in range(len(listLearners)):            
            aQueriedPoint, confidence = listLearners[i].vote()
            if aQueriedPoint in votes:
                votes[aQueriedPoint] += confidence    
            else:
                votes[aQueriedPoint] = confidence
        
        # check if all values are the same
        aKey = next(iter(votes))
        similarValues = all(value == votes[aKey] for value in votes.values())
        if(similarValues):
            a = np.array(list(votes.keys()))
            ask_id = np.random.choice(a, 1)
            ask_id = ask_id[0]  # turn array to scalar
        else:
            ask_id =max(votes)
            
            
                
        theFeature = X[ask_id]
        theLabel = labeler.label(theFeature)        
        trainSet.update(ask_id, theLabel)
       
        
        for l in listLearners:
            l.receive_feedback(ask_id, theLabel)
        
        E_in = np.append(E_in, 1 - baseModel.score(trainSet))
        E_out = np.append(E_out, 1 - baseModel.score(testSet))
    
    
    
    
            
    plt.plot(query_num, E_in, 'g', label='Error in Sample')
    plt.plot(query_num, E_out, 'r', label='Out of Sample')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result Diabetes Dataset')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show()
    
    print (baseModel.score(testSet))

        
        
    
    
    
    
    
    
    
    



#
#class CollaborativeStrategy():        
#    def __init__(self, *args, **kwargs):
#        self.baseModel = kwargs.pop("baseModel", None)
#        
#        if self.baseModel is None:
#            raise TypeError("Proper base exception should be provided")
#        elif  isinstance(self.baseModel, Model):
#            raise TypeError("base model must be an instance of libact.base.interfaces Model class")
#            
#        random_state = kwargs.pop('random_state', None)
#        self.random_state_ = seed_random_state(random_state)
#        
#        self.allLearners = list()
#        
#        
#        models = kwargs.pop('models', None)
#        if models is None:
#            raise TypeError(
#                "__init__() missing required keyword-only argument: 'models'"
#            )
#        elif not models:
#            raise ValueError("models list is empty")
#
#        
#        
#        for model in models:
#            if isinstance(model, str):
#                self.students.append(getattr(libact.models, model)())
#            else:
#                self.students.append(model)
#        self.n_learners = len(self.allLearners)
#        self.teach_students()
#
            
        
