"""
Author:lixing
Data:2020-5-6
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：97%
    运行时长：1025s(训练数据加载14s)
    测试1001s
"""
import numpy as np
from heapq import nsmallest
from basemodel import Model
from basemodel import calTime

class KNN(Model):
	"""docstring for KNN"""
	def __init__(self, K, classnum, trainfile='../mnist_train.csv'):
		super(KNN, self).__init__(trainfile)
		self.K = K
		self.classnum = classnum
	
	@calTime
	def predict(self, x):
		n1 = self.data.shape[0]
		data_data = np.sum(self.data * self.data, axis=-1)
		n2 = x.shape[0]
		predict_all = np.zeros(n2)
		for i in range(n2):
			xi_xi = np.sum(x[i,:] * x[i,:])
			dis = data_data + np.repeat(xi_xi, n1) - 2 * np.dot(self.data, x[i])
			dis = dis.tolist()
			index = list(map(dis.index, nsmallest(self.K, dis)))
			classes = self.label[index]
			predict = np.zeros(self.classnum)
			for c in classes: #变量名一定要谨慎，一开始循环变量为x,导致输入x，变成了一个标量
				predict[c] += 1
			predict_all[i] = np.argmax(predict)

		return predict_all

if __name__ == '__main__':
	knn = KNN(10, 10)
	data, lable = knn.loadData('../mnist_test.csv')
	predict = knn.predict(data)
	accu = knn.accu(predict, lable)
	print('testing accuracy is {}%'.format(accu*100))



