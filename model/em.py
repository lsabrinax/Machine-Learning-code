"""
Author:lixing
Data:2020-5-15
Blog:https://blog.csdn.net/sabrinalx

数据集：伪造数据集,两个高斯分布混合
训练集数量：1000

------------------------------
运行结果：参数顺序依次为alpha mu sigmma_square
producting data parameters are:
[0.3, -2, 0.25, 0.7, 0.5, 1]
predicting parameters are:
[0.3, -2.0, 0.22, 0.70, 0.46, 0.88]
运行时间：0.06s
"""
import numpy as np
import random
import time

class EM(object):
	"""docstring for EM"""
	def __init__(self, w, datasize=1000, epsilon=0.021):
		super(EM, self).__init__()
		self.w = np.array(w)
		self.k = len(w) // 3
		self.datasize = datasize
		self.epsilon = epsilon
		self.data = self.makeData()

	def makeData(self):
		data = []
		for i in range(self.k):
			alpha, mu, sigmma_square = self.w[i * 3], self.w[i * 3 + 1], self.w[i * 3 + 2]
			data.append(np.random.normal(mu, np.sqrt(sigmma_square), int(self.datasize*alpha)))
		data = np.concatenate(data)
		random.shuffle(data)
		self.datasize = data.size
		return data

	def E_step(self, w):
		gamma = []
		for i in range(self.k):
			alpha, mu, sigmma_square = w[i * 3], w[i * 3 + 1], w[i * 3 + 2]
			phi = 1.0 / np.sqrt(2 * np.pi * sigmma_square) * np.exp(-(self.data - mu)**2 / (2 * sigmma_square))
			gamma.append(alpha * phi)
		gamma /= sum(gamma)
		return gamma

	def M_step(self, w, gamma):
		predict = []
		for i in range(self.k):
			mu = w[i * 3 + 1]
			#update alpha
			predict.append(sum(gamma[i]) / len(gamma[i]))
			#update mu
			predict.append(np.dot(self.data, gamma[i]) / sum(gamma[i]))
			#update sigmma_square
			predict.append(np.dot(gamma[i], (self.data - mu)**2) / sum(gamma[i]))
		return predict

	def train(self):
		# predict = np.array([1.0/self.k, 0, 1] * self.k)
		predict = [0.5, 0, 1, 0.5, 1, 1] #参数初始化对结果影响特别大，不同的初始化参数会得到不同的结果
		step = 0
		while(step < 100):
			step += 1
			gamma = self.E_step(predict)
			predict = self.M_step(predict, gamma)
		print('after training parameters became {}'.format(predict))


if __name__ == '__main__':
	start = time.time()
	w = [0.3, -2, 0.25, 0.7, 0.5, 1]
	print('producting data parameters is {}'.format(w))
	em = EM(w)
	em.train()
	print('spending {}s in total'.format(time.time()-start))

