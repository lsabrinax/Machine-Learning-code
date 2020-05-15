# Machine-Learning-code
python实现常见的机器学习算法，实验中用到的数据都为mnist数据，已经转换为csv格式，上传的为数据压缩包，直接解压即可
## 代码结构
所有的的机器学习算法都封装为一个类，都实现了数据预处理、训练和预测这三个接口，然后这些类都继承自basemodel中`Model`这个基类，基类主要负责加载数据和计算准确率
## 目录
**第二章：感知机**  
博客：[机器学习(3)-感知机的理解与代码实现](https://blog.csdn.net/sabrinalx/article/details/105886642)  
代码：[model/perceptron.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/perceptron.py)

**第三章：K近邻**  
博客：[机器学习(5)-K邻近(KNN）的理解与代码实现](https://blog.csdn.net/sabrinalx/article/details/105944938)  
代码：[model/knn.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/knn.py)  

**第四章：朴素贝叶斯**  
博客：[机器学习(2)-朴素贝叶斯的理解和代码实现](https://blog.csdn.net/sabrinalx/article/details/105881335)  
代码：[model/naivebayes.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/naivebayes.py)

**第五章：决策树**   
代码: [model/decision_tree.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/decision_tree.py)  

**第六章：逻辑回归**  
博客：[机器学习(1)-逻辑回归的理解、面试问题以及代码实现](https://blog.csdn.net/sabrinalx/article/details/105875879)  
代码：[model/logistic.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/logistic.py)

**第七章：支持向量机**  
博客：[机器学习(4)-支持向量机的理解与代码实现（上）](https://blog.csdn.net/sabrinalx/article/details/105894364)、[机器学习(4)-支持向量机的理解与代码实现（下）](https://blog.csdn.net/sabrinalx/article/details/105901468)  
代码：[model/svm.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/svm.py)

**第八章：提升方法**
博客：[机器学习(6)-提升方法的理解与代码实现](https://blog.csdn.net/sabrinalx/article/details/105973299)  
代码：[model/adaboost.py](https://github.com/lsabrinax/Machine-Learning-code/blob/master/model/adaboost.py)
