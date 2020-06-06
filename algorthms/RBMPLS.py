#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/12 0012 22:49
#@Author  :    tb_youth
#@FileName:    RBMPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

# import random
# import numpy as np
# import matplotlib.pyplot as plt
from numpy import *
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import neural_network
from sklearn.cross_decomposition import PLSRegression

from algorthms.SplitDataSet import SplitDataHelper

# 一层波尔茨曼机
class One_RBM:
    def __init__(self, n01=5, alpha=0.05, bs=100,
                 ite=100, vb=0, rs=None):
        # 定义一个RBM容器
        self.RBM = neural_network.BernoulliRBM(n_components=n01, learning_rate=alpha, batch_size=bs, n_iter=ite,
                                               verbose=0, random_state=None)

    # 训练RBM 和 SVM两个容器
    def train(self, X):
        # 利用X对RBM容器进行训练
        self.RBM.fit(X)
        # 将训练好的RBM用于对自变量进行特征转换
        X01 = self.RBM.transform(X)
        return X01

    # 测试效果：训练集和测试集都可以用
    def tranform(self, input):
        return self.RBM.transform(input)


# 两层波尔茨曼机
class Two_RBM:
    def __init__(self, n01=8, n02=5, alpha=0.05, bs=100,
                 ite=10, vb=0, rs=None):
        # 定义一个RBM01容器,该容器用于处理前两层
        self.sRBM01 = neural_network.BernoulliRBM(n_components=n01, learning_rate=alpha, batch_size=bs,
                                                  n_iter=ite, verbose=vb, random_state=None)
        # 定义一个RBM02容器，该容器用于处理后两层
        self.sRBM02 = neural_network.BernoulliRBM(n_components=n02, learning_rate=alpha, batch_size=bs,
                                                  n_iter=ite, verbose=vb, random_state=None)

    # 训练RBM
    def train(self, X):
        # step01  利用X对RBM01容器进行训练
        self.sRBM01.fit(X)
        # 将训练好的RBM用于对自变量进行特征转换
        X01 = self.sRBM01.transform(X)
        # step02  利用X01对RBM02容器进行训练
        self.sRBM02.fit(X01)
        X02 = self.sRBM02.transform(X01)
        return X02

    # 测试效果：训练集和测试集都可以用
    def tranform(self, input):
        # 将训练好的RBM用于对自变量进行第一次特征转换
        X01 = self.sRBM01.transform(input)
        # 将训练好的RBM用于对自变量进行第二次特征转换
        X02 = self.sRBM02.transform(X01)
        return X02


class PLS:
    def __init__(self, x0, y0):
        self.r = corrcoef(x0)
        self.m = shape(x0)[1]
        self.n = shape(y0)[1]  # 自变量和因变量个数
        self.row = shape(x0)[0]
        # self.n_components = 0  # 存放最终的成分个数
        # self.coefficient = mat(zeros((self.m, 1)))  # 存放回归系数
        # self.W_star = mat(zeros((self.m, self.m))).T  # 存放W*
        # self.T = mat(zeros((self.row, self.m)))  # 存放成分矩阵
        # self.Ch0 = np.mat(np.zeros((1, self.n)))

    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        return e0, f0

    # 求均值-标准差，回归系数的反标准化时须使用
    def data_Mean_Std(self, x0, y0):
        mean_x = mean(x0, 0)
        mean_y = mean(y0, 0)
        std_x = std(x0, axis=0, ddof=1)
        std_y = std(y0, axis=0, ddof=1)
        return mean_x, mean_y, std_x, std_y

    def Pls(self, x0, y0, h):  # h指的是成分个数
        e0 = mat(x0)
        f0 = mat(y0)
        m = shape(x0)[1]
        ny = shape(y0)[1]
        w = mat(zeros((m, m))).T
        w_star = mat(zeros((m, m))).T
        chg = mat(eye((m)))
        my = shape(x0)[0]
        ss = mat(zeros((m, 1))).T
        t = mat(zeros((my, m)))
        alpha = mat(zeros((m, m)))
        press_i = mat(zeros((1, my)))
        press = mat(zeros((1, m)))
        Q_h2 = mat(zeros((1, m)))
        beta = mat(zeros((1, m))).T
        for i in range(1, m + 1):
            # 计算w,w*和t的得分向量
            matrix = e0.T * f0 * (f0.T * e0)
            val, vec = linalg.eig(matrix)  # 求特征向量和特征值
            sort_val = argsort(val)
            w[:, i - 1] = vec[:, sort_val[:-2:-1]]  # 求最大特征值对应的特征向量
            w_star[:, i - 1] = chg * w[:, i - 1]
            t[:, i - 1] = e0 * w[:, i - 1]
            # temp_t[:,i-1] = t[:,i-1]
            alpha[:, i - 1] = (e0.T * t[:, i - 1]) / (t[:, i - 1].T * t[:, i - 1])
            chg = chg * mat(eye((m)) - w[:, i - 1] * alpha[:, i - 1].T)
            e = e0 - t[:, i - 1] * alpha[:, i - 1].T
            e0 = e
            # 计算ss(i)的值
            # beta = linalg.inv(t[:,1:i-1], ones((my, 1))) * f0
            # temp_t = hstack((t[:,i-1], ones((my,1))))
            # beta = f0\linalg.inv(temp_t)
            # beta = nnls(temp_t, f0)
            beta[i - 1, :] = (t[:, i - 1].T * f0) / (t[:, i - 1].T * t[:, i - 1])  # beta是成分t对y的回归系数，维数和y的维数相同
            cancha = f0 - t * beta  #
            ss[:, i - 1] = sum(sum(power(cancha, 2), 0), 1)  # 注：对不对？？？
            for j in range(1, my + 1):
                if i == 1:
                    t1 = t[:, i - 1]
                else:
                    t1 = t[:, 0:i]
                f1 = f0
                she_t = t1[j - 1, :]
                she_f = f1[j - 1, :]
                k_t1, l_t1 = t1.shape
                k_f1, l_f1 = f1.shape
                t1 = list(t1)
                f1 = list(f1)
                del t1[j - 1]
                del f1[j - 1]  # 删除第j-1个观察值
                # t11 = np.matrix(t1)
                # f11 = np.matrix(f1)
                t1 = array(t1)  # .reshape(k_t1-1, l_t1)
                f1 = array(f1)  # .reshape(k_f1-1, l_f1)
                if i == 1:
                    t1 = mat(t1).T
                    f1 = mat(f1).T
                else:
                    t1 = mat(t1)
                    f1 = mat(f1).T

                beta1 = np.linalg.inv(t1.T * t1) * (t1.T * f1)  # (t[:, i - 1].T * f0) / (t[:, i - 1].T * t[:, i - 1])
                # beta1 = (t1.T * f1) /(t1.T * t1)#error？？？
                cancha = she_f - she_t * beta1
                press_i[:, j - 1] = sum(power(cancha, 2))
            press[:, i - 1] = sum(press_i)
            if i > 1:
                Q_h2[:, i - 1] = 1 - press[:, i - 1] / ss[:, i - 2]
            else:
                Q_h2[:, 0] = 1
            if Q_h2[:, i - 1] < 0.0975:
                h = i
            if h == i:
                break
        return h, w_star, t, beta

    ##计算反标准化之后的系数
    def Calcoef(self, Coef, mean_x, mean_y, std_x, std_y):
        n = len(mean_x)
        n1 = len(mean_y)
        _coef = np.mat(np.zeros((n, n1)))
        ch0 = np.mat(np.zeros((1, n1)))
        for i in range(n1):
            ch0[:, i] = mat(mean_y)[:, i] - mat(std_y)[:, i] * mat(mean_x) / mat(std_x) * Coef[:, i]
            _coef[:, i] = mat(std_y)[0, i] * Coef[:, i] / mat(std_x).T
        return ch0, _coef

    def train(self, x0, y0, h):  # h:成分个数
        # x0, y0 = self.stardantDataSet(x0, y0)  # 先标准化，后面才能反标准化
        mean_x, mean_y, std_x, std_y = self.data_Mean_Std(x0, y0)  # 后面反标准化使用，还有计算可决系数和均方根误差时用
        self.n_components, self.W_star, self.T, beta = self.Pls(x0, y0, h)  # pls算法核心功能

        self.coefficient = self.W_star * beta
        # 反标准化
        # self.Ch0, self.coefficient = self.Calcoef(self.coefficient, mean_x, mean_y, std_x, std_y)
        y_predict = x0 * self.coefficient  # + tile(self.Ch0[0, :], (self.row, 1))
        # 求可决系数和均方根误差
        y_mean = tile(mean_y, (self.row, 1))
        SSE = sum(sum(power((y0 - y_predict), 2), 0))
        SST = sum(sum(power((y0 - y_mean), 2), 0))
        SSR = sum(sum(power((y_predict - y_mean), 2), 0))
        RR = SSR / SST
        RMSE = sqrt(SSE / self.row)
        return y_predict, RR, RMSE  # 训练矩阵的预测值矩阵，训练集的RR和RMSEtrain(self, x0, y0, h)

    def predict(self, x0_te):  # 这个是针对测试集
        # 有点问题，因为这个
        return x0_te * self.coefficient  # + tile(self.Ch0[0, :], (shape(x0_te)[0], 1))  # 预测值矩阵

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


class RMB_PLS:
    def __init__(self, n_components, n01=8, n02=5, alpha=0.05, bs=100, ite=100, vb=0, rs=None):
        self.Two_rbm_model = Two_RBM(n01=n01, n02=n02, alpha=alpha, bs=bs, ite=ite, vb=vb, rs=rs)
        self.pls_model = PLSRegression(n_components=n_components)

    def RMB_train(self, input):
        input_transformed = self.Two_rbm_model.train(input)
        return input_transformed

    def tranform(self, input):
        return self.Two_rbm_model.tranform(input)

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
        self.X_tr_tranformed = self.RMB_train(X_tr)  # X训练集训练
        self.X_te_tranformed = self.tranform(X_te)  # X测试集测试
        self.y_tr_tranformed = self.RMB_train(y_tr)  # y训练集训练
        self.y_te_tranformed = self.tranform(y_te)  # 训练集测试
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

class RunRBMPLS:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict
        self.res_dict = {}


    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.q = parameter_dict.get('q')
        self.n_components = parameter_dict.get('n_components')
        self.n01 = parameter_dict.get('n01')
        self.n02 = parameter_dict.get('n02')
        self.alpha = parameter_dict.get('alpha')
        self.bs = parameter_dict.get('bs')
        self.ite = parameter_dict.get('ite')



    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]
        X = np.mat(X)
        y = np.mat(y)
        # 划分训练集测试集
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X, y, q=self.q)

        # 建模
        """
        RMB_PLS(n_components, n01=8, n02=5, alpha=0.05, bs=100, ite=100, vb=0, rs=None)
        n_components：PLS提取的成分个数，默认为2，可自行设置
        n01：隐层1的神经元个数，默认为8，可自行设置
        n02：隐层2的神经元个数，默认为5，可自行设置
        alpha：学习率，默认为0.05，可自行设置
        bs：batch_size，默认为100，可自行设置
        ite：n_iter，迭代次数，默认100，可行设置
    
        """

        rbm_pls_model = RMB_PLS(self.n_components, n01=self.n01, n02=self.n02,
                                alpha=self.alpha, bs=self.bs, ite=self.ite, vb=0, rs=None)
        y0_tr_predict, y0_tr_RMSE = rbm_pls_model.train(train_x, test_x, train_y, test_y)
        print(y0_tr_RMSE)

        # 预测 test_x
        y0_te_true = rbm_pls_model.y_te_tranformed
        y0_te_predict, y0_te_RMSE = rbm_pls_model.predict()
        print(y0_te_RMSE)

        true_data = pd.DataFrame(y0_te_true)
        predict_data = pd.DataFrame(y0_te_predict)

        show_data_dict = {
            '预测值': predict_data,
            '真实值': true_data
        }

        self.res_dict = {
            '训练集RMSE': y0_tr_RMSE,
            '测试集RMSE': y0_te_RMSE,
            'show_data_dict':show_data_dict
        }

        print(rbm_pls_model.X_te_tranformed.shape)


    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        return self.res_dict


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
        'n01': 8,
        'n02': 5,
        'alpha': 0.05,
        'bs': 100,
        'ite': 100,
    }
    # 设置参数页面
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunRBMPLS(df, all_dict)
    r.run()
