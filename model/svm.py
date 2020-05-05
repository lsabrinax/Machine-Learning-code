"""
Author:lixing
Data:2020-5-4
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000(实际使用：1000)
测试集数量：10000（实际使用：100)
------------------------------
运行结果：
    正确率：92%
    运行时长：26s(数据加载14s)
	训练1.4s,测试0.02s
可以继续优化的地方：
	在算和函数矩阵的时候，一开始用的全部数据，矩阵size太大导致内存不够
	这里可以考虑用LRU机制优化一下，有时间写一下
	如何选择超参数，以及参数的初始化对模型性能的影响
	数据的预处理，如何选择特征，实际问题的抽象
	选择alpha1和alpha2时的策略是否按书上的话效果会更好
"""


import numpy as np
import random
import time
# from functools import wraps
from basemodel import Model
from basemodel import calTime

class SVM(Model):
	"""docstring for SVM"""
	def __init__(self, C=20, trainfile='../mnist_train.csv', sigma=10):
		super(SVM, self).__init__(trainfile)
		self.C = C
		self.sigma = sigma
		self.data, self.label = (d[:1000] for d in self.preprocess(self.data, self.label))
		self.n, self.m = self.data.shape #(samplenum, featurenum)
		self.alpha = np.zeros(self.n)
		# self.alpha = np.random.random(self.n)
		self.b = 0
		self.E = np.zeros(self.n)
		self.g = np.zeros(self.n)
		self.kernel = self.calGaussianKernel(self.data, self.data)
		self.cal_E()

	def calGaussianKernel(self, data1, data2):
		n1 = data1.shape[0]
		n2 = data2.shape[0]
		data1_1 = np.sum(data1 * data1, axis=-1, keepdims=True) # (data1·data2) inner product
		data2_2 = np.sum(data2 * data2, axis=-1, keepdims=True) # (data2·data2)
		data1_2 = np.dot(data1, data2.T) # (data1·data2)
		data = np.tile(data1_1,(1, n2)) + np.tile(data2_2.T, (n1, 1)) - 2*data1_2
		data = np.exp(-data / (2 * self.sigma**2))
		return data

	def cal_E(self):
		self.g = np.dot(self.alpha * self.label, self.kernel) + self.b
		self.E = self.g - self.label

	def IsSatisfyKKT(self, i):
		if self.alpha[i] == 0 and self.label[i]*self.g[i] >= 1:
			return True
		if 0 < self.alpha[i] < self.C and self.label[i]*self.g[i] == 1:
			return True
		if self.alpha[i] == self.C and self.label[i]*self.g[i] <=1:
			return True
		return False

	def getAlpha2(self, i):
		maxdiff = 0
		index = i
		for j in range(self.n):
			if abs(self.E[i] - self.E[j]) > maxdiff:
				maxdiff = abs(self.E[i] - self.E[j])
				index = j
		while(index == i):
			index = random.randint(0, self.n-1)
		return index

	@calTime
	def train(self, maxiter=50): #SMO算法优化过程
		it = 0
		stopiter = False
		while(it < maxiter and not stopiter):
			stopiter = True
			it += 1
			for i in range(self.n):
				if not self.IsSatisfyKKT(i):
					j = self.getAlpha2(i)
					k11 = self.kernel[i][i]
					k22 = self.kernel[j][j]
					k12 = self.kernel[i][j]
					E1 = self.E[i]
					E2 = self.E[j]
					y1 = self.label[i]
					y2 = self.label[j]
					alpha1_old = self.alpha[i]
					alpha2_old = self.alpha[j]
					b_old = self.b
					eta = k11 + k22 - 2 * k12
					alpha2_new = alpha2_old + y2 * (E1 - E2) / eta
					if y1 == y2:
						L = max(0, alpha2_old + alpha1_old - self.C)
						H = min(self.C, alpha2_old + alpha1_old)
					else:
						L = max(0, alpha2_old - alpha1_old)
						H = min(self.C, self.C + alpha2_old - alpha1_old)
					if L == H:
						continue
					if alpha2_new < L:
						alpha2_new = L
					elif alpha2_new > H:
						alpha2_new = H
					alpha1_new = alpha1_old + y1*y2*(alpha2_old - alpha2_new)

					b1_new = -E1 - y1*k11*(alpha1_new - alpha1_old) - y2*k12*(alpha2_new - alpha2_old) + b_old
					b2_new = -E2 - y2*k22*(alpha2_new - alpha2_old) - y1*k12*(alpha1_new - alpha1_old) + b_old

					if 0 < alpha1_new < self.C:
						b_new = b1_new
					elif 0 < alpha2_new < self.C:
						b_new = b2_new
					else:
						b_new = (b1_new + b2_new) / 2
					self.b = b_new
					self.alpha[i] = alpha1_new
					self.alpha[j] = alpha2_new
					self.cal_E()
					if abs(alpha2_new - alpha2_old) >= 0.00001:
						stopiter = False
			print('iterate {}times'.format(it))



	def preprocess(self, data, label):
		"""
		二分类问题，等于零的为1，不等于零为-1,和感知机一样
		特征归一化到[0,1]，不做归一化不改变参数的情况是不行的
		进一步说明SVM对参数和特征的选取还蛮敏感的
		"""
		label[label==0] = -1
		label[label>0] = 1
		
		
		
		
		#不能直接 a /= 255,会报错 TypeError: No loop matching the specified signature and casting was found for ufunc true_divide
		#或者可以先将data转化为float
		data = data / 255 
		return data, label

	@calTime
	def predict(self, data):
		kernel = self.calGaussianKernel(self.data, data)
		g = np.dot(self.alpha * self.label, kernel)
		return np.sign(g)

	
if __name__ == "__main__":
	svm = SVM()
	svm.train()
	testdata, testlabel = svm.loadData('../mnist_test.csv')
	testdata, testlabel = svm.preprocess(testdata[:100], testlabel[:100])
	predict = svm.predict(testdata)
	accu = svm.accu(predict, testlabel)
	print('testing accuracy is {}%'.format(accu*100))