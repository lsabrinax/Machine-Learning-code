import numpy as np
import csv
import time
from functools import wraps

def calTime(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		res = func(*args, **kwargs)
		print('executing {} spends {}s'.format(func.__name__, time.time()-start))
		return res
	return wrapper

class Model(object):
	"""This is a basic model for other machine learning model
	basic function include (1)data loading (2)calculate accuracy
	all subclass should overwrite method "train" and "predict"
	"""
	def __init__(self, trainfile):
		super(Model, self).__init__()
		self.trainfile = trainfile
		self.data, self.label = self.loadData(trainfile)
	

	def loadData(self, path):
		data, label = [], []
		print('start loading data from {}'.format(path))
		start = time.time()
		#method 1 compare two methods time consuming, about same with method 2
		with open(path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				items = line.strip().split(',')
				label.append(int(items[0]))
				data.append(list(map(int,items[1:])))

		#method 2 about 
		# with open(path, 'r') as f:
		# 	reader = csv.reader(f)
		# 	#different model should make different data preprocessing
		# 	#this method just reading origin data is ok
		# 	for line in reader:
		# 		label.append(int(line[0]))
		# 		data.append(list(map(lambda x: int(int(x) >= 128), line[1:]))) #make an experiment binarize for the time being
		print('loading data from {} spends {}s'.format(path, time.time()-start))
		return np.array(data), np.array(label)

	def accu(self, predict, label):
		return sum((predict == label).astype('int')) / len(label)

	def train(self):
		"""
		you should overwrite this method in subclass
		"""
		raise NotImplementedError

	def predict(self, data):
		"""
		you should overwrite this method in subclass
		"""
		raise NotImplementedError

	def preprocess(self, data, label):
		"""
		you should overwrite this method in subclass
		"""
		raise NotImplementedError

