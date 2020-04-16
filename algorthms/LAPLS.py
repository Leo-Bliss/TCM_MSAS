#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :    2020/3/11 0011 20:52
# @Author  :    tb_youth
# @FileName:    LAPLS.py
# @SoftWare:    PyCharm
# @Blog    :    https://blog.csdn.net/tb_youth

# coding:utf-8
from numpy import *
import numpy as np
from sklearn import preprocessing
import itertools
import random
import pandas as pd


# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import pylab
# import numpy as np
# import pandas as pd


# 数据读取-单因变量与多因变量
def loadDataSet01(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    row = len(arrayLines)
    x = mat(zeros((row, 9)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index, :] = curLine[0:9]
        y[index, :] = curLine[-1]
        index += 1
    return x, y


# 数据随机划分
def splitDataSet(x, y, q=0.7):  # q表示训练集的样本占比, x,y不能是DataFrame类型
    m = shape(x)[0]
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


class LAPLS:
    def __init__(self, ntest=20, th_k=0.2, lambd_k=9):
        self.ntest = ntest
        self.th_k = th_k
        self.lambd_k = lambd_k
        # # Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        assert self.th_k > 0.1, 'Warning:Parameter th_k value is not reasonable!'
        assert self.lambd_k > 8, 'Warning:Parameter lambd_k value is not reasonable!'

    # 数据标准化
    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        return e0, f0

    # 求均值-标准差
    def data_Mean_Std(self, x0, y0):
        x0 = mat(x0, dtype=np.float64)
        y0 = mat(y0, dtype=np.float64)
        mean_x = mean(x0, 0)
        mean_y = mean(y0, 0)
        std_x = std(x0, axis=0, ddof=1)
        std_y = std(y0, axis=0, ddof=1)
        return mean_x, mean_y, std_x, std_y

    # PLS核心函数
    def PLS(self, x0, y0):
        e0, f0 = self.stardantDataSet(x0, y0)
        e0 = mat(e0)
        f0 = mat(f0)
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
        h = 0
        for i in range(1, m + 1):
            # 计算w,w*和t的得分向量
            matrix = e0.T * f0 * (f0.T * e0)
            val, vec = linalg.eig(matrix)  # 求特征向量和特征值
            sort_val = argsort(val)
            index_vec = sort_val[:-2:-1]
            w[:, i - 1] = vec[:, index_vec]  # 求最大特征值对应的特征向量
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
            beta[i - 1, :] = (t[:, i - 1].T * f0) / (t[:, i - 1].T * t[:, i - 1])
            cancha = f0 - t * beta
            ss[:, i - 1] = sum(sum(power(cancha, 2), 0), 1)  # 注：对不对？？？
            for j in range(1, my + 1):
                if i == 1:
                    t1 = t[:, i - 1]
                else:
                    t1 = t[:, 0:i]
                f1 = f0
                she_t = t1[j - 1, :]
                she_f = f1[j - 1, :]
                t1 = list(t1)
                f1 = list(f1)
                del t1[j - 1]
                del f1[j - 1]  # 删除第j-1个观察值
                # t11 = np.matrix(t1)
                # f11 = np.matrix(f1)
                t1 = array(t1)
                f1 = array(f1)
                if i == 1:
                    t1 = mat(t1).T
                    f1 = mat(f1).T
                else:
                    t1 = mat(t1)
                    f1 = mat(f1).T

                beta1 = linalg.inv(t1.T * t1) * (t1.T * f1)
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
                break
        return h, w_star, t, beta

    ##计算反标准化之后的系数
    def Calxishu(self, xishu, mean_x, mean_y, std_x, std_y):
        n = shape(mean_x)[1]
        n1 = shape(mean_y)[1]
        xish = mat(zeros((n, n1)))
        ch0 = mat(zeros((1, n1)))
        for i in range(n1):
            ch0[:, i] = mean_y[:, i] - std_y[:, i] * mean_x / std_x * xishu[:, i]
            xish[:, i] = std_y[0, i] * xishu[:, i] / std_x.T
        return ch0, xish

    # Lasso函数
    def lasso_regression(self, X, y, lambd, th):
        '''
            通过坐标下降(coordinate descent)法获取Lasso回归系数
        '''
        # 计算残差平方和
        rss = lambda X, y, w: (y - X * w).T * (y - X * w)
        # 初始化回归系数w.
        m, n = X.shape
        w = mat(zeros((n, 1)))
        r = rss(X, y, w)
        # 使用坐标下降法优化回归系数w
        niter = itertools.count(1)
        for it in niter:
            for k in range(n):
                # 计算常量值z_k和p_k
                # z_k = (X[:, k].T*X[:, k])[0, 0]
                p_k = 0
                for i in range(m):
                    p_k += X[i, k] * (y[i, 0] - sum([X[i, j] * w[j, 0] for j in range(n) if j != k]))
                if p_k < -lambd / 2:
                    w_k = (p_k + lambd / 2) / m
                elif p_k > lambd / 2:
                    w_k = (p_k - lambd / 2) / m
                else:
                    w_k = 0
                w[k, 0] = w_k
            r_prime = rss(X, y, w)
            # a=r
            # AA = abs(r_prime - r)
            delta = abs(r_prime - r)[0, 0]
            r = r_prime
            threshold = th
            # print('Iteration: {}, delta = {}'.format(it, delta))
            if delta < threshold:
                break
        ww = w
        return w

    # 坐标迭代
    def lasso_traj(self, e0, f0, ntest, th_k, lambd_k):
        '''
            获取回归系数轨迹矩阵-即坐标下降迭代
        '''
        _, n = e0.shape
        ws = zeros((ntest, n))
        for i in range(ntest):
            w = self.lasso_regression(e0, f0, lambd=exp(i - lambd_k), th=(th_k - 0.1))  # th取0.1较好
            ws[i, :] = w.T
            # print('lambda = e^({}), w = {}'.format(i-10, w.T[0, :]))
        wwss = ws
        return ws

    # PLS回归系数压缩
    def PLS_xish(self, xish, wss):
        m, n = shape(wss)
        niter = 10
        for i in range(n):
            value = wss[niter, i]
            if value == 0:
                xish[i] = 0
        return xish

    def getRRandRMSE(self, y0, y0_predict):
        row = shape(y0)[0]
        mean_y = mean(y0, 0)
        y_mean = tile(mean_y, (row, 1))
        SSE = sum(sum(power((y0 - y0_predict), 2), 0))
        SST = sum(sum(power((y0 - y_mean), 2), 0))
        SSR = sum(sum(power((y0_predict - y_mean), 2), 0))
        RR = SSR / SST
        RMSE = sqrt(SSE / row)
        return RR, RMSE

    def train(self, x0, y0):
        e0, f0 = self.stardantDataSet(x0, y0)
        mean_x, mean_y, std_x, std_y = self.data_Mean_Std(x0, y0)
        self.m = shape(x0)[1]
        self.n = shape(y0)[1]  # 自变量和因变量个数
        row = shape(x0)[0]

        self.h, self.w_star, self.t, self.beta = self.PLS(x0, y0)
        xishu = self.w_star * self.beta

        # 反标准化
        self.ch0, self._coef = self.Calxishu(xishu, mean_x, mean_y, std_x, std_y)

        ws = self.lasso_traj(e0, f0, self.ntest, self.th_k, self.lambd_k)
        self._coef = self.PLS_xish(self._coef, ws)

        # 求可决系数和均方根误差
        y_tr_predict = x0 * self._coef + tile(self.ch0[0, :], (row, 1))
        y_tr_RR, y0_tr_RMSE = self.getRRandRMSE(y0, y_tr_predict)

        return y_tr_predict, y_tr_RR, y0_tr_RMSE

    def predict(self, x0, y0):  # 其实训练集可测试集都可以用
        row = x0.shape[0]
        y0_predict = x0 * self._coef + tile(self.ch0[0, :], (row, 1))
        y0_RR, y0_RMSE = self.getRRandRMSE(y0, y0_predict)
        return y0_predict, y0_RR, y0_RMSE

    def getSelectedX(self, input):
        # 回归系数为0的特征删除
        row, column = input.shape
        count0 = 0
        for i in range(column):
            if self._coef[i] == 0:
                count0 += 1
        SelectedX = mat(zeros((row, column - count0)))
        j = 0
        for i in range(column):
            if self._coef[i] != 0:
                SelectedX[:, j] = input[:, i]
                j = j + 1
        return SelectedX


class RunLAPLS:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict

    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.q = parameter_dict.get('q')
        self.ntest = parameter_dict.get('ntest')
        self.th_k = parameter_dict.get('th_k')
        self.lambd_k = parameter_dict.get('lambd_k')

    def run(self):
        self.initParameter()
        x0 = np.mat(self.df[self.independent_var])
        y0 = np.mat(self.df[self.dependent_var])

        # 划分训练集测试集
        train_x, train_y, test_x, test_y = splitDataSet(x0, y0, q=self.q)

        # 建模
        """
        LAPLS(ntest=20, th_k=0.2, lambd_k=9)
        ntest = 20 
        th_k = 0.2  # th_k取值得大于0.1，建议精度至4个小数点（最佳取值th_k = 0.2）--默认值0.2
        lambd_k = 9  # lambd_k得大于7，最佳取值lambd_k=9(默认值)
        """

        lapls_model = LAPLS(ntest=self.ntest, th_k=self.th_k, lambd_k=self.lambd_k)
        y_predict, y_RR, y_RMSE = lapls_model.train(train_x, train_y)  # 训练
        _coef = lapls_model._coef
        print(y_RMSE)
        print("回归系数", _coef)

        # 预测训练集
        y_predict, y_RR, y_RMSE = lapls_model.predict(train_x, train_y)
        print("训练集", y_RMSE)

        # 预测测试集
        y_te_predict, y_te_RR, y_te_RMSE = lapls_model.predict(test_x, test_y)
        print("测试集", y_te_RMSE)

        # 经特征选择后的X
        selected_x0 = lapls_model.getSelectedX(x0)
        print(selected_x0)

    def getRes(self):
        pass


# 主函数
if __name__ == '__main__':
    # x0, y0 = loadDataSet01('data/TCMdata.txt')#单因变量与多因变量
    # print(x0)
    # df = pd.read_excel("../data/data02.xlsx")
    # x0 = np.mat(df[['x1','x2','x3','x4','x5','x6','x7','x8','x9']])
    # y0 = np.mat(df[['y']])
    #
    # # 划分训练集测试集
    # train_x, train_y, test_x, test_y = splitDataSet(x0, y0, q=0.8)
    #
    # # 建模
    # """
    # LAPLS(ntest=20, th_k=0.2, lambd_k=9)
    # ntest = 20
    # th_k = 0.2  # th_k取值得大于0.1，建议精度至4个小数点（最佳取值th_k = 0.2）--默认值0.2
    # lambd_k = 9  # lambd_k得大于7，最佳取值lambd_k=9(默认值)
    # """
    # lapls_model = LAPLS(ntest=20, th_k=0.2, lambd_k=9)
    # y_predict, y_RR, y_RMSE = lapls_model.train(train_x, train_y)  # 训练
    # _coef = lapls_model._coef
    # print(y_RMSE)
    # print("回归系数", _coef)
    #
    # # 预测训练集
    # y_predict, y_RR, y_RMSE = lapls_model.predict(train_x, train_y)
    # print("训练集", y_RMSE)
    #
    # # 预测测试集
    # y_te_predict, y_te_RR, y_te_RMSE = lapls_model.predict(test_x, test_y)
    # print("测试集", y_te_RMSE)
    #
    # # 经特征选择后的X
    # selected_x0 = lapls_model.getSelectedX(x0)
    # print(selected_x0)

    df = pd.read_excel("../data/data02.xlsx")
    var_dict = {
        'independ_var': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
        'depend_var': ['y']
    }
    parameter_dict = {
        'q': 0.8,
        'ntest': 20,
        'th_k': 0.2,
        'lambd_k': 9
    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunLAPLS(df, all_dict)
    r.run()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # xx = arange(0, 10.0, 0.02)
    # ax.plot(y_predict, 'r:', markerfacecolor='blue', marker='o')
    # plt.annotate('y_predict', xy=(6, 0.090), xytext=(4, 0.10),
    #              arrowprops=dict(facecolor='black', shrink=0.05))
    # # plt.title('y_predict')
    #
    # # ax = fig.add_subplot(112)
    # ax.plot(y0, markerfacecolor='red', marker='h')
    # # plt.title('y0')
    # plt.annotate('y0', xy=(7.2, 0.058), xytext=(8, 0.05),
    #              arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.grid(True)
    # plt.show()
    # parameter_Errors(th_k, lambd_k)
