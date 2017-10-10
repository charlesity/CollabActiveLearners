#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:15:22 2017

@author: charley
"""
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
from libact.utils import inherit_docstring_from, seed_random_state, zip



class Alearner:
    def __init__(self, queryStrategy, hisModel, hisDataset):
        if not isinstance(queryStrategy, QueryStrategy):
            raise TypeError("Query strategy must be of type libact.base.interfaces QueryStrategy")    
    
        self.queryStrategy = queryStrategy      
        self.history = []           
        self.accuracy = None
        self.hisBaseModel = hisModel
        self.hisDataset= hisDataset
        self.hisBaseModel.train(self.hisDataset)        
        
    def vote(self):
        queryIndex = self.queryStrategy.make_query()
        if self.accuracy is None:
            return [queryIndex, 1/(self.hisDataset.len_unlabeled())]
        else:   
            return [queryIndex, self.accuracy]      #previous accuracy
  
    def receive_feedback(self, voted_index_x, label):        
        Xdata, _ = zip(*self.hisDataset.data)        
        sample = Xdata[voted_index_x]
        predicted = self.hisBaseModel.predict(sample)
        self.history.append([voted_index_x, label, self.hisBaseModel.predict(sample)[0]])
        self.update_confidence()
        self.hisDataset.update(voted_index_x, label)        
             
        

    def update_confidence(self):  
        self.accuracy = np.sum([v[1]==v[2] for v in self.history])/len(self.history)
     


        
        
        
        
        
        