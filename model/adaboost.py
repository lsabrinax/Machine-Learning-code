"""
Author:lixing
Data:2020-5-15
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000(实际使用1000)
测试集数量：10000(实际使用100)
------------------------------
运行结果：
    正确率：96%
    运行时长：221s(训练数据加载8s)
    训练209s
"""
import numpy as np
from basemodel import Model 
from basemodel import calTime

class AdaBoost(Model):
	"""docstring for AdaBoost"""
	def __init__(self, treenum=20, trainfile='../mnist_train.csv'):
		super(AdaBoost, self).__init__(trainfile)
		self.treenum = treenum
		self.trainfile = trainfile
		self.data, self.label = [x[:1000] for x in self.preprocess(self.data, self.label)]
		self.n, self.m = self.data.shape
		self.trees = [] #产生的多个分类函数
		self.W = [1.0 / self.n] * self.n #权重
		self.div = np.arange(int(np.min(self.data)), int(np.max(self.data)+2)) - 0.5 #分类阈值
		self.rule = ['lessOne', 'moreOne']
		



	def cal_E_Gx(self, L, H, div, f):
		e = 0.0
		Gx = np.zeros(self.n)
		for i in range(self.n):
			if self.data[i][f] < div:
				Gx[i] = L
				if L != self.label[i]:
					e += self.W[i]
			else:
				Gx[i] = H
				if H != self.label[i]:
					e += self.W[i]
		return Gx, e

	@calTime
	def train(self):
		predict = np.zeros(self.n)
		for i in range(self.treenum):
			tree = {'e': 1}
			for r in self.rule:
				if r == 'lessOne':
					L, H = 1, -1
				else:
					L, H = -1, 1
				for d in self.div:
					for i in range(self.m):
						#计算误差率和Gm(x)的输出
						Gx, e = self.cal_E_Gx(L, H, d, i)
						if e < tree['e']:
							tree['div'] = d 
							tree['rule'] = r
							tree['feature'] = i
							tree['e'] = e 
							tree['Gx'] = Gx
			e = tree['e']
			#计算alpha
			alpha = 1 / 2 * np.log((1-e) / e)
			tree['alpha'] = alpha
			Gx = tree['Gx']
			#更新权重
			self.W = self.W * np.exp(-alpha * self.label * Gx)
			self.W /= sum(self.W)
			predict += alpha * Gx
			self.trees.append(tree)
			if sum((np.sign(predict) != self.label).astype('int')) == 0:
				break
		print('total tree num is {}'.format(len(self.trees)))

	@calTime
	def predict(self, data):
		predict = np.zeros(data.shape[0])
		for t in self.trees:
			r = t['rule']
			if r == 'lessOne':
				L, H = 1, -1
			else:
				L, H = -1, 1
			f = t['feature']
			d = t['div']
			alpha = t['alpha']
			predict[data[:,f]<d] += L * alpha
			predict[data[:,f]>=d] += H * alpha
		return np.sign(predict)




	def preprocess(self, data, label):
		data[data<128] = 0
		data[data>=128] = 1
		label[label>0] = 1
		label[label==0] = -1
		return data, label

if __name__ == '__main__':
	adaboost = AdaBoost()
	adaboost.train()
	data, label = adaboost.loadData('../mnist_test.csv')
	data, label = adaboost.preprocess(data[:100], label[:100])
	predict = adaboost.predict(data)
	accu = adaboost.accu(predict, label)
	print('testing accuracy is {}'.format(accu * 100))
