import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler

from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


from Util import *




def load_pickkle(data_root):
	pickle_file = os.path.join(data_root, 'notMNIST.pickle')
	
	trainSet, trainLabel=[], []
	with open(pickle_file, 'rb') as f:
		saved = pickle.load(f)
		trainSet= saved['train_dataset']
		trainLabel = saved['train_labels']
		test_dataset = saved['test_dataset']
		test_labels =saved['test_labels']
		valid_dataset=saved['valid_dataset']
		valid_labels=saved['valid_labels']
			
	return trainSet, trainLabel, test_dataset, test_labels, valid_dataset, valid_labels

def main():
	trn_X, trn_y, test_X,test_y,_,_=load_pickkle(".")
	image_size = 28
	num_of_labels = 100


	# np.savetxt('labels', trn_y, delimiter=',')
	print (np.shape(trn_X), np.shape(trn_y) )

	print(trn_y)
	# trn_X_formated, trn_y_formated = reformat_dataset(trn_X,trn_y, image_size, num_of_labels)

	# test_X_formated, test_y_formated = reformat_dataset(test_X,test_y, image_size, num_of_labels)

	# dataset = Dataset(trn_X_formated, trn_y)
	# test_dataset = Dataset(test_X_formated, test_y)
	# model = LogisticRegression()

	# model.train(dataset)

	# print (model.score(test_dataset))


	# trn_ds=prepare_unlabeled_dataset(trn_X,trn_y, num_of_labels)
	# trn_ds2 = copy.deepcopy(trn_ds)
	# fully_labeled_trn_ds = prepare_full_labeled_set(trn_X, trn_y)

	# lbr = IdealLabeler(fully_labeled_trn_ds)

	# quota= len(trn_y) - 10
	# qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
	# model = LogisticRegression()
	# E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

	# qs2 = RandomSampling(trn_ds2)
	# model = LogisticRegression()
	# E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)


	# query_num = np.arange(1, quota + 1)
	# plt.plot(query_num, E_in_1, 'b', label='qs Ein')
	# plt.plot(query_num, E_in_2, 'r', label='random Ein')
	# plt.plot(query_num, E_out_1, 'g', label='qs Eout')
	# plt.plot(query_num, E_out_2, 'k', label='random Eout')
	# plt.xlabel('Number of Queries')
	# plt.ylabel('Error')
	# plt.title('Experiment Result')
	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
 #               fancybox=True, shadow=True, ncol=5)
	# plt.show()






if __name__ == '__main__':
	main()