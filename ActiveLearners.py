
# from __future__ import division
# import os
# import copy
from multiprocessing import Process, Manager
# import time

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cross_validation import train_test_split

# # libact classes
# from libact.base.dataset import Dataset
# from libact.models import LogisticRegression
# from libact.query_strategies import UncertaintySampling, RandomSampling
# from libact.labelers import InteractiveLabeler
# from libact.labelers import IdealLabeler



def aLabeler(d, l):
    # d[1] = '1'
    # d['2'] = 2
    # d[0.25] = None
    l.reverse()
    # if(d[1]==None):
    # 	something=0
    # d[1] ='1'
    # for key in d:
    # 	# theLabel=labeler.label(X[key])
    #     d[key]=1
        


    # from time import sleep
    # from os import getpid

    # labeler = IdealLabeler(fully_labeled_trn_ds)
    # X, _ = zip(*fully_labeled_trn_ds.data)


def f(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()
    if(d[1]==None):
    	something=0

if __name__ == '__main__':
    manager = Manager()

    d = manager.dict()
    l = manager.list(range(10))

    p1 = Process(target=aLabeler, args=(d, l))
    # p2 = Process(target=aLabeler, args=(d, l))
    p1.start()
    # p2.start()

    
    p1.join()
    # p2.join()

    print d
    print l  
# if __name__ == '__main__':
#     # n_labeled = 5 
#     # n_classes = 2
#     # num_processes =5
#     # digits = load_digits(n_class=n_classes)  # consider binary case
#     # X = digits.data
#     # y = digits.target
#     # quota= len(X)/(len(X)*num_processes)
#     # genDataSet = Dataset(X, y)
    
#     m = Manager()
#     bag = m.dict()
    
#     fully_labeled_trn_ds = None
#     aLabeler= Process(target=aLabeler, args=(fully_labeled_trn_ds, bag)) 
#     aLabeler.start()
    
    # for np in range(num_processes):
    # aLeanerProcess1 = Process(target=aLearner, args=(n_classes, bag, genDataSet, .70,n_labeled,0,0,0)) 
    # aLeanerProcess1.start()
    
    # aLeanerProcess2 = Process(target=aLearner, args=(n_classes, bag, genDataSet, .60,n_labeled,0,1,0)) 
    # aLeanerProcess2.start()
    
    # aLeanerProcess3 = Process(target=aLearner, args=(n_classes, bag, genDataSet, .70,n_labeled,0,0,0)) 
    # aLeanerProcess3.start()


    # aLeanerProcess4 = Process(target=aLearner, args=(n_classes, bag, genDataSet, .60,n_labeled,0,1,0)) 
    # aLeanerProcess4.start()


    
    # aLeanerProcess1.join() 
    # aLeanerProcess2.join()  
    # aLeanerProcess3.join()  
    # aLeanerProcess4.join()    
    # aLabeler.join()
        
        
 