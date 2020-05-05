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
    运行时长：35s(训练数据加载14s)
    训练10s,测试0.04s
"""
import numpy as np
from basemodel import Model
from basemodel import calTime

class LogisticRegression(Model):
	"""docstring for LogisticRegression"""
	def __init__(self, lr=1, max_iter=100, trainfile='../mnist_train.csv'):
		super(LogisticRegression, self).__init__(trainfile)
		self.n, self.m = self.data.shape #(samplenum, featurenum)
		self.weight = np.zeros(self.m+1)
		self.max_iter = max_iter
		self.lr = lr
		self.data, self.label = self.preprocess(self.data, self.label)

	@calTime
	def train(self):
		data = np.concatenate((self.data, np.ones((self.n, 1))), axis=-1)
		label = self.label
		for i in range(self.max_iter):
			predict = self.sigmoid(np.dot(data, self.weight))
			error = label - predict #对应梯度求导公式中真实值和预测值的差
			self.weight += self.lr * np.dot(data.T, error)


	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))

	@calTime
	def predict(self, x):
		n, m = x.shape
		x = np.concatenate((x, np.ones((n, 1))), axis=-1)
		predict = self.sigmoid(np.dot(x, self.weight))
		predict[predict>=0.5] = 1
		predict[predict<0.5] = 0
		return predict

	def preprocess(self, data, label):
		"""
		把十分类问题当作一个二分类问题，这里没有修改判别阈值效果也蛮好
		大于0的label都为1，等于0label为零
		特征归一化到[0,1],不做归一化也ok
		"""
		label[label>0] = 1
		data = data / 255
		return data, label


if __name__ == '__main__':
	lr = LogisticRegression()
	lr.train()
	testdata, testlable = lr.loadData('../mnist_test.csv')
	testdata, testlable = lr.preprocess(testdata, testlable)
	predict = lr.predict(testdata)
	accu = lr.accu(predict, testlable)
	print('testing accuracy is {}%'.format(accu * 100))