"""
Author:lixing
Data:2020-5-15
Blog:https://blog.csdn.net/sabrinalx

数据集：Mnist
训练集数量：60000
测试集数量：10000(实际使用100)
------------------------------
运行结果（ID3,未剪枝）：
    正确率：85%
    运行时长：219s(训练数据加载8s)
    创捷决策树：206s
"""
import numpy as np
import time
from basemodel import Model
from basemodel import calTime

class TreeNode(object):
	"""docstring for TreeNode"""
	def __init__(self, val):
		super(TreeNode, self).__init__()
		self.val = val #如果是叶子节点就保存类别值，如果不是就保存特征index
		self.left = None
		self.right = None

class DecisionTree(Model):
	"""docstring for DecisionTree"""
	def __init__(self, classnum=10, trainfile='../mnist_train.csv', featurenum=784, epsilon=0.1):
		super(DecisionTree, self).__init__(trainfile)
		self.classnum = classnum
		self.trainfile = trainfile
		self.epsilon = epsilon
		self.data, self.label = self.preprocess(self.data, self.label)
		self.root = None


	def calH_D(self, x):
		"""
		x:当前待分类的数据的label
		"""
		size = x.size
		labels = set([label for label in x])
		H_D = 0.0
		for label in labels:
			p = x[x==label].size / size
			H_D += -p * np.log2(p)
		return H_D

	def calH_D_A(self, dataInFeature, labels):
		"""
		dataInFeature:当前待分类样本的某一个特征的具体取值
		labels:当前待分类样本的label
		"""
		H_D_A = 0.0
		#特征取值为零，这里是因为特征取值只有0和1,所以可以这么做
		#如果有超过两个以上的特征就要依次遍历所有特征可能的取值
		size_0 = dataInFeature[dataInFeature==0].size #命名不能以数字开头，例如0_size
		size_1 = dataInFeature[dataInFeature==1].size
		size = dataInFeature.size
		if size_0:
			H_D_A += size_0 / size * self.calH_D(labels[dataInFeature==0])
		if size_1:
			H_D_A += size_1 / size * self.calH_D(labels[dataInFeature==1])
		return H_D_A


	def selectBestF(self, data, x, feature):
		"""
		data:当前待分类样本的特征值(所有的特征值)
		x:对应的待分类样本的label
		"""
		maxG_D_A = -1
		selectedF = -1
		for f in feature:
			H_D = self.calH_D(x)
			G_D_A = H_D - self.calH_D_A(data[:,f], x)
			if G_D_A > maxG_D_A:
				maxG_D_A = G_D_A
				selectedF = f 
		#这样写会有个问题，例如，建立左子树的时候把特征a删除了，但是右子树可能也是通过特征a可以得到最大信息增益
		# self.feature.remove(selectedF) 
		return maxG_D_A, selectedF


	def majorClass(self, labels):
		count = [0] * self.classnum
		for l in labels:
			count[l] += 1
		return count.index(max(count))

	# @calTime
	def createTree(self, data, label, feature):
		# print('{} data to be divided by {} features'.format(label.size, len(self.feature)))
		slabel = set([x for x in label])
		if len(slabel) == 1:
			return TreeNode(label[0]) #set 不能通过索引访问 所以slabel[0]会报错


		if len(feature) == 0:
			return TreeNode(self.majorClass(label))


		G_D_A, selectedF = self.selectBestF(data, label, feature)

		if G_D_A < self.epsilon:
			return TreeNode(self.majorClass(label))

		feature.remove(selectedF)

		root = TreeNode(selectedF)
		root.left = self.createTree(data[data[:,selectedF]==0], label[data[:,selectedF]==0], feature[:])
		root.right = self.createTree(data[data[:,selectedF]==1], label[data[:,selectedF]==1], feature[:])
		return root

	@calTime
	def predict(self, data):
		size = data.shape[0]
		predict = np.zeros(size)
		for i in range(size):
			predict[i] = self.binarySearch(data[i,:])
		return predict

	def binarySearch(self, data):
		cur = self.root
		while(cur.left and cur.right):
			if data[cur.val] == 0:
				cur = cur.left
			else:
				cur = cur.right
		return cur.val

	def preprocess(self, data, label):
		data[data<128] = 0
		data[data>=128] = 1
		return data, label

if __name__ == "__main__":
	dtree = DecisionTree()
	start = time.time()
	dtree.root = dtree.createTree(dtree.data, dtree.label, list(range(784)))
	print('creating tree spends {}s'.format(time.time() - start))
	data, label = dtree.loadData('../mnist_test.csv')
	data, label = dtree.preprocess(data[:100], label[:100])
	predict = dtree.predict(data)
	accu = dtree.accu(predict, label)
	print('testing accuracy is {}%'.format(accu * 100))
