"""
Author:lixing
Data:2020-5-4
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：99%
    运行时长：124s(训练数据加载14s)
    训练100s,测试0.05s
"""


import numpy as np
import time
from basemodel import Model
from basemodel import calTime

class Perceptron(Model):
	"""docstring for Perceptron"""
	def __init__(self, trainpath='../mnist_train.csv', featurenum=784, lr=0.001, maxiter=5):
		super(Perceptron, self).__init__(trainpath)
		self.maxiter = maxiter
		self.weight = np.zeros(featurenum+1) #[w, b]
		self.lr = lr
		self.data, self.label = self.preprocess(self.data, self.label)

	@calTime
	def train(self):
		sampleNum, featureNum = self.data.shape
		data = np.concatenate((self.data, np.ones((sampleNum, 1))), axis=-1)
		label = self.label
		#because cannot promise sample totally linear separable，set maxiter to stop training
		for i in range(self.maxiter):
			#every step choose one sample to update weight
			for j in range(sampleNum):
				if label[j] * sum((data[j] * self.weight)) <= 0:
					self.weight += self.lr * label[j] * data[j] #update weight
	@calTime
	def predict(self, data):
		n, m = data.shape
		data = np.concatenate((data, np.ones((n, 1))), axis=-1)
		predict = np.dot(data, self.weight)
		return np.sign(predict)

	def preprocess(self, data, label):
		"""
		二分类问题，等于零的为-1，不等于零为1
		特征归一化到[0,1],不归一化也是可以的，在没有改变参数的情况下
		准确率为97.4%，这个调整参数应该就可以达到一样的结果
		"""
		label[label==0] = -1
		label[label>0] = 1
		data = data / 255
		return data, label 

if __name__ == "__main__":
	perception = Perceptron()
	perception.train()
	data, label = perception.loadData('../mnist_test.csv')
	data, label = perception.preprocess(data, label)
	predict = perception.predict(data) 
	accu = perception.accu(predict, label)
	print('testdata accuracy is {}%'.format(accu * 100))
