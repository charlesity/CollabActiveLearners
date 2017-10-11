#!/usr/bin/env python3
"""
This script simulates real world use of active learning algorithms. Which in the
start, there are only a small fraction of samples are labeled. During active
learing process active learning algorithm (QueryStrategy) will choose a sample
from unlabeled samples to ask the oracle to give this sample a label (Labeler).

In this example, ther dataset are from the digits dataset from sklearn. User
would have to label each sample choosed by QueryStrategy by hand. Human would
label each selected sample through InteractiveLabeler. Then we will compare the
performance of using UncertaintySampling and RandomSampling under
LogisticRegression.
"""
from __future__ import division
import os
from os import getpid
import copy
import multiprocessing as mp
from multiprocessing import Process, Manager, Lock
import time
from time import sleep
import types

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import InteractiveLabeler
from libact.labelers import IdealLabeler
import matplotlib.pyplot as plt

def samplingTechniqueSelector(techniqueIndex, theDataSet, theModelIndex):
    return {
    0:  UncertaintySampling(theDataSet, method='lc', model=modelSelector(theModelIndex)),
    1:  UncertaintySampling(theDataSet, method='sm', model=modelSelector(theModelIndex))
    }.get(techniqueIndex, UncertaintySampling(theDataSet, method='lc', 
        model=modelSelector(theModelIndex)))

def modelSelector(modelIndex):
    return {
    0:  LogisticRegression()
    }.get(modelIndex, LogisticRegression())

def aLabeler(fully_labeled_trn_ds, theBag, theValue, theLock):
    # 
    # labeler = IdealLabeler(fully_labeled_trn_ds)
    # X, _ = zip(*fully_labeled_trn_ds.data)
    # # 
    totalLabelsProvided = 0
    # while (theValue.value > 0):
    #     for key in theBag.keys():
    #         if(theBag[key] == -1):
    #             # label it
    #             try:
    #                 theLabel = labeler.label(X[key])
    #                 theBag[key]= theLabel
    #                 totalLabelsProvided +=1
    #             except Exception as e:
    #                 print (e)
    #             #
    # print ('Labeler:Num of learners with queries',theValue.value)
    # print ('Total Labels provided ', totalLabelsProvided)
# 
def aLearner(n_classes,theBag, theValue, theLock, generalDataset, hisDataQuota, hisTotalLabeledY, hisModelIndex, sampleTechniqueIndicator, sampleTechniqueModelIndicator):
    np.random.seed(getpid())    
    randomNum= np.random.random_integers(low=0,high=100)
    
    trainX, trainY= zip(*generalDataset.data)
    hisTrainX, hisTestX, hisTrainY, hisTestY = train_test_split(trainX, trainY, train_size=hisDataQuota, random_state=randomNum)
    # 
    while(len(np.unique(hisTrainY[:hisTotalLabeledY]))<n_classes):
        hisTrainX, hisTestX, hisTrainY, hisTestY = train_test_split(trainX, trainY, train_size=hisDataQuota,random_state=randomNum)
    # 
    # 
    hisTrainingDataset= Dataset(hisTrainX, 
        np.concatenate([hisTrainY[:hisTotalLabeledY], [None] * (len(hisTrainY) - hisTotalLabeledY)]))
    hisTestDataset = Dataset(hisTestX, hisTestY)
    # 
    hisModel=modelSelector(hisModelIndex)
    hisModel.train(hisTrainingDataset)

    hisSamplingTechnique = samplingTechniqueSelector(samplingTechniqueSelector, hisTrainingDataset
        , sampleTechniqueModelIndicator)
    # 
    E_in, E_out = [], []
    numOfQueries = 0
    while(numOfQueries < 20):
        print('progress')
        # try:
        ask_id= hisSamplingTechnique.make_query()
        # sleep(2)
        # obtain true id from generalDataset
        X_Data, _ = zip(*generalDataset.data)
        for true_index in range(len(X_Data)):
            if (hisTrainX[ask_id]==X_Data[true_index]).all():
                break


        if  theBag[true_index] is not None:
            if theBag[true_index] != -1:
                hisTrainingDataset.update(ask_id, theBag[true_index])
                hisModel.train(hisTrainingDataset)                
                numOfQueries=numOfQueries + 1
                E_in = np.append(E_in, 1 - hisModel.score(hisTrainingDataset))
                E_out = np.append(E_out, 1 - hisModel.score(hisTestDataset))
        else:            
            theBag[true_index] = -1 # mark it to request for the label
        # except Exception as e:
        #     theValue.value = theValue.value - 1
        #     print('exception caught', e)
    theValue.value = theValue.value - 1 # done quering oracle    
    print ('Process ID:',getpid(),'Num of learners with queries',theValue.value)
    # 
    # Individual performance
    print (E_in)
    print (E_out)
    query_num = np.arange(1, numOfQueries + 1)
    plt.plot(query_num, E_in, 'b', label='Process E_in:'+str(getpid()))
    plt.plot(query_num, E_out, 'r', label='Process E_out:'+str(getpid()))
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result'+'Process:'+str(getpid()))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    plt.show()

  
# 
if __name__ == '__main__':
    from sklearn.datasets import load_digits  
    n_labeled = 5 
    n_classes = 4
    num_processes = 1
    digits = load_digits(n_class=n_classes)  # consider binary case
    X = digits.data
    y = digits.target
    quota= len(X)/(len(X)*num_processes)
    genDataSet = Dataset(X, y)
    processes=[]
    fully_labeled_trn_ds = Dataset(X, y)
    with Manager() as m:
        try:
            bag = m.dict()
            l = m.Value('i',num_processes)
            theLock=m.Lock()        
            for index in range(len(X)):
                bag[index] = None
            bag['predict']=False
            # aLabeler = Process(target=aLabeler, args=(fully_labeled_trn_ds, bag, l, theLock))
            # aLabeler.start()
            

            for nProcess in range(num_processes):
                aLeanerProcess = Process(target=aLearner, args=(n_classes, bag,l, theLock, genDataSet, .70,n_labeled,0,0,0))
                aLeanerProcess.start()
                processes.append(aLeanerProcess)


            labeler = IdealLabeler(fully_labeled_trn_ds)
            X, _ = zip(*fully_labeled_trn_ds.data)
            # 
            totalLabelsProvided = 0
            while (l.value > 0):
                for key in bag.keys():
                    if(bag[key] == -1):
                        # label it
                        try:
                            theLabel = labeler.label(X[key])
                            bag[key]= theLabel
                            totalLabelsProvided +=1
                        except Exception as e:
                            print (e)
                        #
            print ('Labeler:Num of learners with queries',l.value)
            print ('Total Labels provided ', totalLabelsProvided)


            # 
            # while True:
            #     if not mp.active_children():
            #         break;
            for aProcess in reversed(processes):
                aProcess.join()
        except Exception as e:
            print (e)





        
 