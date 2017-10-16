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
from libact.base.interfaces import QueryStrategy, Model
from libact.query_strategies import UncertaintySampling, RandomSampling, QUIRE, HintSVM
import libact.models
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.labelers import IdealLabeler

from alearner import Alearner

LOGGER = logging.getLogger(__name__)


class CollaborativeStrategy(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(CollaborativeStrategy, self).__init__(*args, **kwargs)
        self.baseModel = kwargs.pop('baseModel',None)
        if self.baseModel is None:
            raise ValueError("Collaborative Strategy requires a base model")
        self.queryStrategies = kwargs.pop('queryStrategies', None)
        if self.queryStrategies is None or not isinstance(self.queryStrategies, list):
            raise ValueError("Collaborative Strategy requires a list of Query strategies ")
        self.baseModel.train(self.dataset)
        self.aLearners = list()
        for qs in self.queryStrategies:
            a1 = Alearner(qs, hisModel = SVM(kernel='linear', decision_function_shape='ovr'), hisDataset= copy.deepcopy(self.dataset))
            self.aLearners.append(a1)
        
        
        self.baseModel.train(self.dataset)
    
    
    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        trainSet.update(ask_id, theLabel)
        self.send_feedback(ask_id, theLabel)
    
    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        X, _ = zip(*self.dataset.data)
        votes =  {}
        for i in range(len(self.aLearners)):            
            aQueriedPoint, weigted_confidence = self.aLearners[i].vote()
            if aQueriedPoint in votes:
                votes[aQueriedPoint] += weigted_confidence
            else:
                votes[aQueriedPoint] = weigted_confidence
        
        # check if all values are the same
        aKey = next(iter(votes))
        similarValues = all(value == votes[aKey] for value in votes.values())
        if(similarValues):
            a = np.array(list(votes.keys()))
            ask_id = np.random.choice(a, 1)
            ask_id = ask_id[0]  # turn array to scalar
        else:
            ask_id =max(votes)
        return ask_id
    
    def send_feedback(self, ask_id, theLabel):
        for l in listLearners:
            l.receive_feedback(ask_id, theLabel)

        

        
        
    
    
   