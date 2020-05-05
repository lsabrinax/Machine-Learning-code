"""
Mnist classification by NaiveBayes
Author:lixing
Data:2020-5-4
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：84.3%
    运行时长：26s(训练数据加载14s)
    训练0.6s,测试1.7s
"""

import numpy as np
import time
from basemodel import Model
from basemodel import calTime

class NaiveBayes(Model):
	"""docstring for NaiveBayes"""
	def __init__(self, featurenum, classnum,trainfile='../mnist_train.csv'):
		super(NaiveBayes, self).__init__(trainfile)
		self.featurenum = featurenum
		self.classnum = classnum
		self.py = np.zeros(classnum)
		self.py_x = np.zeros((classnum, featurenum, 2))
		self.data, self.label = self.preprocess(self.data, self.label)

	@calTime
	def train(self):
		"""
		对先验概率和条件概率都取了对数，这样之后的连乘就可以改为加法
		这里计算先验概率和条件概率的方法用的都是书中说的贝叶斯估计
		"""
		data = self.data
		label = self.label
		n, m = data.shape
		#calculate priori probability p(y),用每一类的样本数除以总的样本数
		for i in range(self.classnum):
			self.py[i] = np.log((sum((label==i).astype('int')) + 1)/ (n + self.classnum))

		#计算条件概率p(x|y),循环遍历每一个类别，对每一个类别的data，在样本数量方向求和
		#因为每个特征都只有0和1两个取值，求和就得到了特征取1的样本数
		for i in range(self.classnum):
			self.py_x[i,:,1] = (np.sum(data[label==i], axis=0) + 1)  / (sum((label==i).astype('int')) + 2)
		self.py_x[:,:,0] = 1 - self.py_x[:,:,1]#用1减去类别为1的概率就得到了零的概率
		self.py_x = np.log(self.py_x)

	@calTime
	def predict(self, data):
		"""
		data: np.array (n, m)
		"""
		n, m = data.shape
		prob = np.zeros((n, self.classnum))
		featureindex = list(range(self.featurenum))
		# py_x = np.tile(self.py_x, (n, 1, 1, 1))
		for j in range(n):
			#减少了一层循环，速度就提高了大约25倍，原来要50s,现在测试只需要2秒
			prob[j] = np.sum(self.py_x[:,featureindex, data[j]], axis=-1) + self.py
			# for i in range(self.classnum):
			# 	# prob[j][i] = sum(self.py_x[i,list(range(self.featurenum)),data[j]]) + self.py[i]
			# 	# prob[j][i] = sum(self.py_x[i,featureindex,data[j]]) + self.py[i] #和上面一行相比时间就快了两秒，看来主要是循环造成的

		index = prob.argmax(axis=-1) #返回每一行最大值的索引
		return index

	def preprocess(self, data, label):
		"""
		是分类问题，特征数为784
		对特征值做二值化处理，相当于每个特征只有两种取值
		"""
		data[data<128] = 0
		data[data>=128] = 1
		return data, label




if __name__ == '__main__':
	bayes = NaiveBayes(784, 10)
	bayes.train()
	testdata, testlabel = bayes.loadData('../mnist_test.csv')
	testdata, testlabel = bayes.preprocess(testdata, testlabel)
	predict = bayes.predict(testdata) 
	accu = sum((predict == testlabel).astype('int')) / len(testlabel)
	print('test accuracy is {}%'.format(accu * 100)) #84.3%