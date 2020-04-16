#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/12 0012 23:15
#@Author  :    tb_youth
#@FileName:    SEAPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


"""
自编码器中加入稀疏限制
当加入稀疏性限制后，我们将重点关注的是中间层的输出，而分析导致其激活的影响因子
加入稀疏性就是为了获得超完备基
容易欠拟合
"""
#---导入库包---

import numpy as np
from sklearn.cross_decomposition import PLSRegression

import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
import random  # 这个包要在下面，不然会被numpy中的random方法覆盖，会报错
random.seed(0)
#---产生一个随机矩阵---
def rand(a, b):
    return 2*np.random.random((a,b)) - 1

#---使用s函数---
def s(x,deriv = False):
    x = np.array(x, dtype=np.float64)
    if(deriv==True):
        return x*(1-x)
    return 1.0/(1+np.exp(-x))

def splitDataSet(x, y, q=0.7):  # q表示训练集的样本占比, x,y不能是DataFrame类型
    m =shape(x)[0]
    train_sum = int(round(m * q))
    # 利用range()获得样本序列
    randomData = range(0, m)
    randomData = list(randomData)
    # 根据样本序列进行分割- random.sample(A,rep)
    train_List = random.sample(randomData, train_sum)
    test_List = list(set(randomData).difference(set(train_List)))
    # 获取训练集数据-train
    train_x = x[train_List, :]
    train_y = y[train_List, :]
    # 获取测试集数据-test
    test_x = x[test_List, :]
    test_y = y[test_List, :]
    return train_x, train_y, test_x, test_y
#---定义一个自联想网络---
class SAE:
    #---该神经网络包括：输入层，隐含层，输出层---
    def __init__(self, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05):
        #---定义网络的值---
        self.ny = ny
        self.iterations = iterations
        self.beta = beta
        self.eta = eta
        self.sp = sp

    #---正向传播---
    def forward(self, inputs, ni, no):  # ni, no输入输出层的神经元个数，等于inputs的特征个数
        # ---生成权重矩阵---
        self.w1 = rand(ni + 1, self.ny)
        self.w2 = rand(self.ny + 1, no)
        #---阈值b---
        b = np.ones((len(inputs),1))
        #---Step00 原始数据 -> 输入层---
        self.ai = np.c_[np.array(inputs),b]
        #---Step01 输入层 -> 隐含层---
        self.ah = s(np.dot(self.ai , self.w1))
        self.ay = np.c_[s(np.dot(self.ai , self.w1)),b]
        #---Step02 隐含层 -> 输出层---
        self.ao = s(np.dot(self.ay , self.w2))
    #---反向传播---
    def backward(self , targets):
        #---数据的样本数维度---
        mi = len(targets)
        #---Step00 真实输出值---
        self.at = np.array(targets)
        #---加入稀疏性规则项---
        rho = (1.0/mi)*sum(self.ah,0)
        #---计算KL散度---
        Jsparse = sum(self.sp*np.log(self.sp/rho)+(1-self.sp)*np.log((1-self.sp)/(1-rho)))
        #---稀疏规则项的偏导---
        sterm = self.beta*(-self.sp/rho+(1-self.sp)/(1-rho))
        #---Step01 计算输出层的梯度---
        o_deltas = s(self.ao , True) * (self.at - self.ao)
        #---Step02 计算隐含层的梯度,这里是加入了稀疏导项---
        y_deltas = s(self.ay , True) * (np.dot(o_deltas , self.w2.T)\
        -np.c_[np.tile(sterm,(mi,1)),np.zeros((mi,1))])
        #---Step03 更新权值---
        self.w2 = self.w2 + (self.eta*1.0)*(np.dot(self.ay.T , o_deltas))
        self.w1 = self.w1 + (self.eta*1.0)*(np.dot(self.ai.T , y_deltas[:,:-1]))
        #---计算代价J(w,b)给予输出显示---
        Jcost = sum((0.5/mi)*sum((self.at - self.ao)**2))
        return Jcost +self.beta*Jsparse
    #---训练函数---
    def train(self, patterns):
        inputs = np.array(patterns)
        error = 0.0
        liust = list()
        plt.figure()
        for k in range(self.iterations):
            self.forward(inputs, inputs.shape[1], inputs.shape[1])  # 后面两个参数是输入输出神经元的个数，等于inputs的特征个数
            error = self.backward(inputs)
            liust.append(error)
            if k % 100 == 0:
                print(error)
        xh = range(self.iterations)
        plt.plot(xh, liust, 'r')
    #---取出中间层---
    def tranform(self , iput):
        # 阈值b
        b = np.ones((len(iput),1))
        # Step00 原始数据 -> 输入层
        ai = np.c_[np.array(iput),b]
        # Step01 输入层 -> 隐含层
        ay = s(np.dot(ai , self.w1))
        return ay

class SAE_PLS:
    def __init__(self, n_components, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05):
        self.pls_model = PLSRegression(n_components=n_components)
        self.sae_model = SAE(ny=ny, iterations=iterations, beta=beta, eta=eta, sp=sp)

    def SAE_train(self, input):
        self.sae_model.train(input)  # 训练
        input_transformed = self.sae_model.tranform(input)
        return input_transformed

    def tranform(self, input):
        return self.sae_model.tranform(input)

    def PLS_train(self, x0_tr, y0_tr):
        # 训练模型，预测训练集
        self.pls_model.fit(x0_tr, y0_tr)
        # 取出一些结果，比如成分个数，回归系数这些
        self.coefficient = self.pls_model.coef_
        y0_tr_predict = self.pls_model.predict(x0_tr)
        y0_tr_RR, y0_tr_RMSE = self.RRandRMSE(y0_tr, y0_tr_predict)
        # print("y0_tr_RR", y0_tr_RR)
        # y1_tr_RR = self.pls_model.score(x0_tr, y0_tr)
        # print("y1_tr_RR", y1_tr_RR)
        return y0_tr_predict, y0_tr_RR, y0_tr_RMSE

    def PLS_predict(self, x0_te, y0_te):  # 其实训练集可测试集都可以用
        # 预测测试集，!!! 不行，如果没分训练集和测试集呢(还没想好怎么写),这里应该是这样写
        y0_te_predict = self.pls_model.predict(x0_te)
        y0_te_RR, y0_te_RMSE = self.RRandRMSE(y0_te, y0_te_predict)
        return y0_te_predict, y0_te_RR, y0_te_RMSE

    def train(self, X_tr, X_te, y_tr, y_te):
        # 受限玻尔兹曼机RBM映射后X，Y
        self.X_tr_tranformed = self.SAE_train(X_tr)  # X训练集训练、映射
        self.X_te_tranformed = self.tranform(X_te)  # X测试集映射
        self.y_tr_tranformed = self.SAE_train(y_tr)  # y训练集训练、映射
        self.y_te_tranformed = self.tranform(y_te)  # 训练集映射
        # PLS回归
        y0_tr_predict, y0_tr_RR, y0_tr_RMSE = self.PLS_train(self.X_tr_tranformed, self.y_tr_tranformed)
        return y0_tr_predict, y0_tr_RMSE

    def predict(self):  # 只能预测测试集
        y0_te_predict, y0_te_RR, y0_te_RMSE = self.PLS_predict(self.X_te_tranformed, self.y_te_tranformed)
        return y0_te_predict, y0_te_RMSE


    # 求可决系数和均方根误差
    def RRandRMSE(self, y0, y0_predict):  # 这个是针对测试集
        row = shape(y0)[0]
        mean_y = mean(y0, 0)
        y_mean = tile(mean_y, (row, 1))
        SSE = sum(sum(power((y0 - y0_predict), 2), 0))
        SST = sum(sum(power((y0 - y_mean), 2), 0))
        SSR = sum(sum(power((y0_predict - y_mean), 2), 0))
        RR = SSR / SST
        RMSE = sqrt(SSE / row)
        return RR, RMSE



# 运行算法接口
class RunSEAPLS:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict

    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.q = parameter_dict.get('q')
        self.n_components = parameter_dict.get('n_components')
        self.ny = parameter_dict.get('ny')
        self.iterations = parameter_dict.get('iterations')
        self.beta = parameter_dict.get('beta')
        self.eta = parameter_dict.get('eta')
        self.sp = parameter_dict.get('sp')

    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]
        X = np.mat(X)
        y = np.mat(y)
        # print(X)
        # print(y)

        # 划分训练集测试集
        train_x, train_y, test_x, test_y = splitDataSet(X, y, q=self.q)

        # 建模
        """
        SAE_PLS(n_components, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
        n_components：PLS提取的成分个数，默认为2，可自行设置
        ny：隐含层神经元个数，一般大于输入层的神经元个数，默认22，可自行设置
        iterations：迭代次数，默认为1000，可自行设置
        beta：代价函数中的一个压缩参数，默认0.5，可自行设置
        eta：步长，相当于学习率，默认为1，可自行设置
        sp：稀疏性参数，通常是一个接近于0的较小值，默认为0.05，可自行设置
        """
        sae_pls_model = SAE_PLS(self.n_components, ny=self.ny, iterations=self.iterations,
                                beta=self.beta, eta=self.eta, sp=self.sp)
        y0_tr_predict, y0_tr_RMSE = sae_pls_model.train(train_x, test_x, train_y, test_y)
        print("训练集", y0_tr_RMSE)

        # 预测 test_x
        y0_te_predict, y0_te_RMSE = sae_pls_model.predict()
        print("测试集", y0_te_RMSE)

        print(sae_pls_model.X_te_tranformed.shape)

    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        pass



#---测试---
if __name__ == '__main__':
    # 读取数据
    df = pd.read_excel("../data/data01.xlsx")

    # 变量字典：自变量，因变量
    var_dict = {
        'independ_var': ["x1", "x2", "x3", "x4"],
        'depend_var': ["y"]
    }
    # 参数字典，需要设置的参数
    parameter_dict = {
        'q': 0.8,
        'n_components': 2,
        'ny': 22,
        'iterations': 1000,
        'beta': 0.5,
        'eta': 1,
        'sp': 0.05
    }
    # 设置参数页面
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunSEAPLS(df, all_dict)
    r.run()

    # 读取数据
    # data_df = pd.read_excel("data/data01.xlsx")
    # X = data_df[["x1", "x2", "x3", "x4"]]
    # y = data_df.y.values.reshape(len(X), 1)
    #
    # X = np.mat(X)
    # y = np.mat(y)
    # print(X)
    # print(y)
    #
    # # 划分训练集测试集
    # train_x, train_y, test_x, test_y = splitDataSet(X, y, q=0.8)
    #
    # # 建模
    # """
    # SAE_PLS(n_components, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    # n_components：PLS提取的成分个数，默认为2，可自行设置
    # ny：隐含层神经元个数，一般大于输入层的神经元个数，默认22，可自行设置
    # iterations：迭代次数，默认为1000，可自行设置
    # beta：代价函数中的一个压缩参数，默认0.5，可自行设置
    # eta：步长，相当于学习率，默认为1，可自行设置
    # sp：稀疏性参数，通常是一个接近于0的较小值，默认为0.05，可自行设置
    # """
    # sae_pls_model = SAE_PLS(2, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    # y0_tr_predict, y0_tr_RMSE = sae_pls_model.train(train_x, test_x, train_y, test_y)
    # print("训练集", y0_tr_RMSE)

    # # 预测 test_x
    # y0_te_predict, y0_te_RMSE = sae_pls_model.predict()
    # print("测试集", y0_te_RMSE)
    #
    # print(sae_pls_model.X_te_tranformed.shape)

    # pat = [[0,0, 0,0],
    #     [0,1, 0,1],
    #     [1,0, 1,0],
    #     [1,1, 1,1]]
    # ab = n.SAE_train(np.array(pat))
    # ah = n.tranform(np.array(pat))
    # print(ab)
    # print(ah)
    # print(ah.shape)

    # n = SAE(ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    # n.train(np.array(pat))
    # ah=n.tranform(np.array(pat))
    # print(ah)
    # print(ah.shape)
