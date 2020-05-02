#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/22 0022 16:44
#@Author  :    tb_youth
#@FileName:    PLSSDA.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
# from sklearn.metrics import classification_report
from sklearn import preprocessing
import pandas as pd

class Softmax:
    def __init__(self, alpha=0.001, maxCycles=150):
        self.reg = 0.0005
        self.alpha = alpha  # 梯度下降的步长
        self.maxCycles = maxCycles  # 迭代次数

    # 计算损失函数和梯度
    def loss_dw(self, W, train_x, train_y):  # reg为正则化惩罚系数
        loss = 0.0  # 初始化损失值
        dW = np.zeros_like(W)  # 初始化梯度值，权重W所对应的梯度
        m = train_x.shape[0]
        s = np.dot(train_x, W)  # dot为矩阵相乘,该步是计算指数，s的格式为（样本数,类别数）。每个样本都和每个类别的自变量权值相乘，最终得到每个样本属于每个类别的概率。
        scores = s - np.max(s, axis=1)  # s中的每个数据减去每列的最大值,（样本数,类别数）
        scores_E = np.exp(scores)  # （样本数,类别数）
        Z = np.sum(scores_E, axis=1)
        prob = scores_E / Z
        y_trueClass = np.zeros_like(prob)
        # y_trueClass[range(m),train_y]=1.0
        # 取出y_trueClass所指向的正确类的概率值
        for i in range(m):  # 创建指示性函数矩阵，此处的类别必须是连续的。
            j = int(train_y[i, 0])
            y_trueClass[i, j] = 1.0
        # print('y_trueClass:',y_trueClass)
        loss += -np.sum(y_trueClass * np.log(prob).T) / m + 0.5 * self.reg * np.sum(
            W * W)  # 求最小代价函数，log下什么都没默认底数为e，其他情况为：log2,log10。+号后为L2正则化或者是权重衰减项，目的：1、不仅避免数据上溢。2、这个权重衰减项以后，代价函数就变成了严格的凸函数，这样就可以保证得到唯一的解了。此时的Hessian矩阵变为可逆矩阵，并且因为是凸函数，梯度下降法和L-BFGS等算法可以保证收敛到全局最优解。
        dW += -np.dot(train_x.T, y_trueClass - prob) / m + self.reg * W  # 计算的梯度，此处用了L1正则化
        return loss, dW

    # 梯度下降更新W，使损失函数最小
    def gradDescent(self, W, train_x, train_y):
        dW = np.zeros_like(W)  # 初始化梯度值，权重W所对应的梯度
        all_loss = []
        # all_dW=[]
        DE_loss, dW = self.loss_dw(W, train_x, train_y)
        min_loss = DE_loss
        for i in range(self.maxCycles):
            W = W - self.alpha * dW
            DE_loss, dW = self.loss_dw(W, train_x, train_y)
            all_loss.append(DE_loss)
            # all_dW.append(dW)
            if DE_loss <= min_loss:
                min_loss = DE_loss
            else:
                break
        # print('W:',W)
        return W, min_loss, i, all_loss

    # 预测部分
    def predict(self, test_x, W):
        # m=np.shape(test_x)[0]
        # test_x=np.hstack((test_x, np.mat(np.ones((m,1)))))##在test_x后面加上一列全为1的值，，即和偏置项相乘
        s = np.dot(test_x, W)
        scores = s - np.max(s, axis=1)  # s中的每个数据减去每列的最大值,（样本数,类别数）
        scores_E = np.exp(scores)  # （样本数,类别数）
        Z = np.sum(scores_E, axis=1)
        pre_prob = scores_E / Z  # 每个样本属于每个类的概率
        pre_y = np.argmax(pre_prob, axis=1)  # 返回每行最大值的索引，即输出预测类别
        return pre_prob, pre_y

class PLS_S_DA:
    def __init__(self, n_class, h=1, alpha=0.001, maxCycles=150):
        self.lam = 0.0001
        self.h = h  # PLS成分个数
        self.n_class = n_class  # 类别数
        self.softmax = Softmax(alpha=alpha, maxCycles=maxCycles)

    # 数据标准化
    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        e0 = np.mat(e0)
        f0 = np.mat(f0)
        return e0, f0

    ##获取矩阵的行列数
    def row_col(self, x):
        m, n = np.shape(x)
        return m, n

    ##对划分数据进行0均值标准化处理
    ##原数据类型是mat
    def standardDataSetZscoreMat(self, x, y):
        m, n = self.row_col(x)
        n1 = np.shape(y)[1]
        sx = np.array(np.ones((m, n)))
        sy = np.array(np.ones((m, n1)))
        ##求取x的每一列的均值
        mean_x = np.mean(x, 0)  ##0表示最终结果得到的是1*n的形式
        # print("mean_x", mean_x, "\n")
        # ---0代表行，1表示列（我这么写是为了便于理解）
        ##axis=0代表对每一列求，ddof=1 意味着分母除以n-1，ddof=0，意味着分母除以n
        std_x = np.std(x, axis=0, ddof=1)
        # print("std_x",std_x)
        sx = (x - np.tile(mean_x, (m, 1))) / std_x
        ##等价于sx = (x-mean_x)/std_x
        # print(sx[1:5,:])
        mean_y = np.mean(y, 0)  ##0表示行，1表示列
        std_y = np.std(y, axis=0, ddof=1)
        sy = (y - mean_y) / std_y
        # print("mean_y",mean_y)
        return sx, sy

    ##参数初始化
    def init(self, m, n):
        return np.mat(np.zeros((m, n)))

    ##计算特征值和特征向量
    def eig_vec_val(self, x):
        ##此时得到的特征值和特征向量并没有按照从大到小的顺序排序
        eigvalue, eigvector = np.linalg.eig(x)
        return eigvalue, eigvector

    ##得到矩阵最大特征值对应的特征向量
    def ascend_eig_vec_val(self, x):
        eigvalue, eigvector = self.eig_vec_val(x)
        '''
        #sorted_indices[:-k-1:-1]则利用切片的语法特性，保留了前K大的特征值对应的下标。
        切片有三个参数[start : end : step]，当step为-1时，表示逆序，从最后一个元素开始，
        一直到第end+1个元素为止。sorted_indices[:-k-1:-1]则表示从最后一个元素一直到第k个
        为止的所有下标，也就是前k大的特征值对应的下标。
        '''

        sorted_indices = np.argsort(eigvalue)  ##返回的是逆序的下标值，从小到大排序（三个参数[start : end : step]）
        # print (sorted_indices)
        top_evecs = eigvector[:, sorted_indices[:-2:-1]]  ##取最大特征值对应的特征向量
        # print("top_evecs=")
        # print(np.shape(top_evecs))
        return top_evecs

    ##求逆矩阵，如果可逆，直接求，不可逆，求
    def dsingular(self, x, lam):  ##如果
        ##逆矩阵必须为方阵
        m = np.shape(x)[0]
        if np.linalg.det(x) == 0:
            return np.linalg.inv(x + lam * np.eye(m))
        else:
            return np.linalg.inv(x)

    def PLS(self, E0, F0, n):  ##n需要提取的主成分数
        # E0 = np.mat(E0)
        # F0 = np.mat(F0)
        m, mx = np.shape(E0)  ##训练集样本数、主成分个数,dimendion
        my = np.shape(F0)[1]
        # print(my)
        new_train_x = np.hstack((np.mat(np.ones((m, 1))), E0))  ##python矩阵连接方式
        t = self.init(m, n)
        v = self.init(my, n)
        u = self.init(m, n)
        p = self.init(mx, n)
        r = self.init(my, n)
        q = self.init(my, n)
        chg = np.mat(np.eye(mx))
        w = self.init(mx, n)
        w_ = self.init(mx, n)
        xishu = self.init(mx, my)
        xishu1 = self.init(mx, my)
        raw_E0 = E0
        raw_F0 = F0
        for i in range(n):  ##n<=mx
            ##求第一主成分
            # print(E0.T * F0*F0.T *E0)
            # temp_mat = E0.T * F0 * F0.T * E0
            # print(temp_mat.shape)
            w[:, i] = self.ascend_eig_vec_val(E0.T * F0 * F0.T * E0)
            ##currentLine = list(map(np.float64, currentLine))##映射成float类型
            w_[:, i] = chg * w[:, i]  ##计算w*
            t[:, i] = E0 * w[:, i]  # 求自变量的主成分
            # print t
            ##得到载荷向量矩阵
            p[:, i] = E0.T * t[:, i] * self.dsingular((t[:, i].T * t[:, i]), self.lam)  ##载荷向量
            r[:, i] = F0.T * t[:, i] * self.dsingular((t[:, i].T * t[:, i]), self.lam)  ##载荷向量
            # print(np.shape(r))
            ## 求取F0的主成分
            v[:, i] = self.ascend_eig_vec_val(F0.T * E0 * E0.T * F0)
            u[:, i] = F0 * v[:, i]  ##求因变量的主成分
            ##求参数
            q[:, i] = F0.T * u[:, i] * self.dsingular((u[:, i].T * u[:, i]), self.lam)
            ####更新chg，为下一次计算w_做准备
            chg = chg * (np.mat(np.eye((mx)) - w[:, i] * p[:, i].T))
            ##得到残差矩阵
            E1 = E0 - t[:, i] * p[:, i].T
            F1 = F0 - u[:, i] * r[:, i].T
            ##求系数
            xishu = w_[:, i] * (r[:, i].T)
            xishu1 = xishu1 + w_[:, i] * (r[:, i].T)
            # sum_res=sum_res+ raw_E0*xishu
            # print("raw_E0*xishu=")
            # print(raw_E0*xishu)
            # print ("sum_res","\n")
            # print(sum_res)
            # residual =Euclid(sum_res,raw_F0)
            # residual2 =Euclid(raw_E0* xishu1, raw_F0)
            # print ("xishu-》累计求和residual=")
            # print(residual)
            # print ("xishu1-》residual2=")
            # print(residual2)
            E0 = E1
            F0 = F1
        # print("最终的系数1-》xishu1=")
        # print(xishu1)##
        b1 = self.dsingular((t.T * t), self.lam) * (t.T * raw_F0)  ##最终系数Y =E * b2->Y= t*b1->E*w *b1
        b2 = w_ * b1  ##w*b
        # residual1 = np.sum(np.sum(np.power(t*b1 - raw_F0, 2), 0), 1)
        # print("最小二乘t算出来的系数residual1=")
        # print(residual1)
        # print("t")
        # print(t)
        return b2, xishu1, t, u, w

    ##求均值
    def CalMean(self, x, y):  ####0表示行，1表示列
        return np.mean(x, 0), np.mean(y, 0)

        ##求标准差

    # ---0代表行，1表示列（我这么写是为了便于理解）
    ##axis=0代表对每一列求，ddof=1 意味着分母除以n-1，ddof=0，意味着分母除以n
    def CalVal(self, x, y):
        return np.std(x, axis=0, ddof=1), np.std(y, axis=0, ddof=1)

    def Calxishu(self, b, xishu1, mean_x, mean_y, std_x, std_y, n, n1):
        ##n1是矩阵y的列数
        b1 = np.mat(np.zeros((n, n1)))
        b2 = np.mat(np.zeros((1, n1)))
        for i in range(n1):  ##求斜率
            # a1[:,i]=b[:,i]* std_y/std_x.T
            b1[:, i] = xishu1[:, i] * std_y / std_x.T
        for i in range(n1):  ##求偏置项
            # a2[:,i]=mean_y - std_y*mean_x/ std_x * b[:,i]
            b2[:, i] = mean_y - std_y * mean_x / std_x * xishu1[:, i]
        # rb2 = np.vstack((a1,a2))
        rxishu1 = np.vstack((b1, b2))
        return rxishu1

    def train(self, train_x, train_y):
        train_x = np.mat(train_x)
        train_y = np.mat(train_y)
        n = train_y.shape[1]
        m = train_x.shape[1]
        train_x1, train_y1 = self.stardantDataSet(train_x, train_y)  # Z-score标准化
        # train_x1 = np.mat(train_x1)
        # train_y1 = np.mat(train_y1)
        b2, xishu1, t, u, w = self.PLS(train_x1, train_y1, self.h)  # h为提取主成分的个数，lam初始值为0.0001
        # print t
        W = np.random.randn(self.h, self.n_class) * 0.0001  # W的格式为（自变量数，类别数）
        for j in range(n):  # range()括号里面的值与因变量的个数一致
            W, min_loss, i, all_loss = self.softmax.gradDescent(W, t, train_y[:, j])
            xishu1 = w * W
            # print ('xishu1:',np.shape(xishu1))
            mean_x, mean_y = self.CalMean(train_x, train_y)  ##0表示最终结果得到的是1*n的形式
            std_x, std_y = self.CalVal(train_x, train_y)
            self._coef = self.Calxishu(b2, xishu1, mean_x, mean_y, std_x, std_y, m, self.n_class)  # 3为类别数
            # print ('rxishu:',np.shape(rxishu))
        pre_probj, pre_yj, mean_accuracy = self.predict(train_x, train_y)  # 得到预测以及准确率
        return self._coef, pre_yj, mean_accuracy  # 回归系数，训练集的预测值，多个因变量的平均准确率（一般是单因变量）

    def Accuracy_score(self, true_y, pre_y):
        accuracy = []
        for j in range(true_y.shape[1]):
            accuracy_i = accuracy_score(true_y[:, j], pre_y)  # 一个因变量的accuracy
            accuracy.append(accuracy_i)
        mean_accuracy = np.mean(accuracy, axis=0)  ##多个因变量的accuracy平均值
        return mean_accuracy

    def predict(self, test_x, test_y):
        m = np.shape(test_x)[0]
        test_x = np.hstack((test_x, np.mat(np.ones((m, 1)))))  ##在test_x后面加上一列全为1的值，，即和偏置项相乘
        s = np.dot(test_x, self._coef)
        scores = s - np.max(s, axis=1)  # s中的每个数据减去每列的最大值,（样本数,类别数）
        scores_E = np.exp(scores)  # （样本数,类别数）
        Z = np.sum(scores_E, axis=1)
        pre_prob = scores_E / Z  # 每个样本属于每个类的概率
        pre_y = np.argmax(pre_prob, axis=1)  # 返回每行最大值的索引，即输出预测类别
        mean_accuracy = self.Accuracy_score(test_y, pre_y)  # 代表多个因变量的平均准确率
        return pre_prob, pre_y, mean_accuracy  # 每个样本属于每个类的概率, 预测类别, 多个因变量的平均准确率
 # 运行算法接口
class RunPLSSDA:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict
        self.res_dict = {}

    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.h = parameter_dict.get('h')
        self.alpha = parameter_dict.get('alpha')
        self.maxCycles = parameter_dict.get('maxCycles')
        self.n_splits = parameter_dict.get('n_splits')


    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]

        all_accuracy = []  # 存放测试集每一折的准确率
        n_class = len(np.unique(y))  # 类别数
        pls_s_da = PLS_S_DA(n_class, h=self.h, alpha=self.alpha, maxCycles=self.maxCycles)
        """
        PLS_S_DA算法是分类算法，只接受单因变量
        PLS_S_DA(n_class, h=1, alpha=0.001, maxCycles=150)
        n_class：类别数，因数据集不同而不同n_class = len(np.unique(y))
        h：成分个数，默认1，可自行设置
        alpha：梯度下降的步长，相当于学习率，默认0.001，可自行设置
        maxCycles：最大迭代次数，默认150，可自行设置
        """

        kf = KFold(n_splits=self.n_splits, shuffle=True)
        # kf = KFold(atrrmat.shape[0],n_folds=10,shuffle=True)
        fold = 0
        for train_index, test_index in kf.split(X):
            fold += 1
            train_x, test_x = X.values[train_index], X.values[test_index]
            train_y, test_y = y.values[train_index], y.values[test_index]
            train_x = np.mat(train_x,dtype=int)
            train_y = np.mat(train_y,dtype=int)
            test_x = np.mat(test_x,dtype=int)
            test_y = np.mat(test_y,dtype=int)
            n = train_y.shape[1]
            m = train_x.shape[1]

            # 训练
            _coef, pre_train_y, accuracy_train_y = pls_s_da.train(train_x, train_y)  # 回归系数，训练集的预测值，准确率（单因变量）
            # 测试
            pre_prob, pre_test_y, accuracy_test_y = pls_s_da.predict(test_x, test_y)  # 每个样本属于每个类的概率, 预测类别, 准确率
            all_accuracy.append(accuracy_test_y)

        mean_allaccuracy = np.mean(all_accuracy)  # 十折的平均准确率

        print('all_accuracy:', all_accuracy)
        print('mean_allaccuracy:', mean_allaccuracy)
        self.res_dict = {
            'all_accuracy':all_accuracy,
            'mean_allaccuracy':mean_allaccuracy
        }


    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        return self.res_dict


if __name__ == '__main__':
    # 读取数据集
    df = pd.read_csv('../data/balance-scale.csv')  # 导入数据
    headers = df.columns.values.tolist()
    # X = df[headers[1:]]
    # y = df[headers[0:1]]
    var_dict = {
        'independ_var': headers[1:],
        'depend_var': headers[0:1]
    }
    parameter_dict = {
        'h': 1,
        'alpha': 0.001,
        'maxCycles': 150,
        'n_splits':10

    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunPLSSDA(df, all_dict)
    r.run()
    # all_accuracy = []  # 存放测试集每一折的准确率
    # n_class = len(np.unique(y))  # 类别数
    # pls_s_da = PLS_S_DA(n_class, h=1, alpha=0.001, maxCycles=150)
    # """
    # PLS_S_DA算法是分类算法，只接受单因变量
    # PLS_S_DA(n_class, h=1, alpha=0.001, maxCycles=150)
    # n_class：类别数，因数据集不同而不同n_class = len(np.unique(y))
    # h：成分个数，默认1，可自行设置
    # alpha：梯度下降的步长，相当于学习率，默认0.001，可自行设置
    # maxCycles：最大迭代次数，默认150，可自行设置
    # """
    # kf = KFold(n_splits=10, shuffle=True)
    # # kf = KFold(atrrmat.shape[0],n_folds=10,shuffle=True)
    # fold = 0
    # for train_index, test_index in kf.split(X):
    #     fold += 1
    #     train_x, test_x = X.values[train_index], X.values[test_index]
    #     train_y, test_y = y.values[train_index], y.values[test_index]
    #     train_x = np.mat(train_x)
    #     train_y = np.mat(train_y)
    #     test_x = np.mat(test_x)
    #     test_y = np.mat(test_y)
    #     n = train_y.shape[1]
    #     m = train_x.shape[1]
    #
    #     # 训练
    #     _coef, pre_train_y, accuracy_train_y = pls_s_da.train(train_x, train_y) # 回归系数，训练集的预测值，准确率（单因变量）
    #     # 测试
    #     pre_prob, pre_test_y, accuracy_test_y = pls_s_da.predict(test_x, test_y)  # 每个样本属于每个类的概率, 预测类别, 准确率
    #     all_accuracy.append(accuracy_test_y)
    #
    # mean_allaccuracy = np.mean(all_accuracy)  # 十折的平均准确率
    #
    # print('all_accuracy:', all_accuracy)
    # print('mean_allaccuracy:', mean_allaccuracy)


