#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/16 0016 21:01
#@Author  :    tb_youth
#@FileName:    RFPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

import numpy as np
from numpy import *
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn import preprocessing
from algorthms.SplitDataSet import SplitDataHelper


class RF_PLS:
    def __init__(self, h, n_estimators=10, max_depth=None):
        self.h = h
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    # 数据标准化
    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        return e0, f0

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

    # 随机森林做回归
    def rf_Regressor(self, t, y):  # 求成分t对y的回归
        model = RandomForestRegressor(n_estimators=10)
        model.fit(t, y)
        y_pre = model.predict(t)  # 预测y
        return y_pre

    # PLS核心函数
    def PLS(self, x0, y0):
        e0 = mat(x0)
        f0 = mat(y0)
        f0_original = f0  # 原始f0，求残差时使用
        m = shape(x0)[1]
        ny = shape(y0)[1]
        w = mat(zeros((m, self.h)))
        w_star = mat(zeros((m, self.h)))
        chg = mat(eye((m)))
        my = shape(x0)[0]
        ss = mat(zeros((m, 1))).T
        t = mat(zeros((my, self.h)))  # (n, h)
        alpha = mat(zeros((m, self.h)))
        # press_i = mat(zeros((1, my)))
        # press = mat(zeros((1, m)))
        # Q_h2 = mat(zeros((1, m)))
        # beta = mat(zeros((1, m))).T
        h = self.h + 1
        for i in range(1, h):
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
            # beta[i - 1, :] = (t[:, i - 1].T * f0) / (t[:, i - 1].T * t[:, i - 1])
            # cancha = f0 - t * beta
            if i == 1:
                t_ = t[:, i - 1]
            else:
                t_ = t[:, 0:i]
            # print("===============================", t_.shape)  # 出现了两次一维(已改)
            self.model.fit(t_, f0)  # 用随机森林计算成分t对y的回归（建模）
            f0 = f0_original - self.model.predict(t_)  # 求残差（预测）
            # ss[:, i - 1] = sum(sum(power(cancha, 2), 0), 1)  # 注：对不对？？？
        return w_star, t#, beta

    def train(self, x0, y0):
        x0 = mat(x0, dtype=np.float64)
        y0 = mat(y0, dtype=np.float64)
        self.m = shape(x0)[1]
        self.n = shape(y0)[1]  # 自变量和因变量个数
        row = shape(x0)[0]

        self.w_star, self.t = self.PLS(x0, y0)

        # 求可决系数和均方根误差
        y_tr_predict = self.model.predict(self.t)  # 仅针对于训练集进行预测，因为这里的t是训练集得到的
        y_tr_RR, y0_tr_RMSE = self.getRRandRMSE(y0, y_tr_predict)

        return y_tr_predict, y_tr_RR, y0_tr_RMSE

    def predict(self, x0, y0):  # 可预测训练集和测试集
        x0 = mat(x0)
        y0 = mat(y0)
        # 先根据w_star矩阵求出t矩阵, 再由t预测y
        t = x0 * self.w_star
        y0_predict = self.model.predict(t)
        y0_RR, y0_RMSE = self.getRRandRMSE(y0, y0_predict)
        return y0_predict, y0_RR, y0_RMSE

class RunRFPLS:
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
        self.h = parameter_dict.get('h')
        self.n_estimators = parameter_dict.get('n_estimators')
        self.max_depth = parameter_dict.get('max_depth')


    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X.values, y.values.reshape(X.shape[0], 1), q=self.q)

        # 步骤:3：建模
        """
        RF_PLS(h, n_estimators=10, max_depth=None)
        h = 10  # 成分个数，默认为10，其应小于等于样本的维度，从前台传进来应注意
        n_estimators = 10  # 随机森林中的树的数量，默认为10，可自行设置
        max_depth = None  # 树的最大深度， 默认None，节点被扩展，直到所有叶子都是纯的（可自行设置）
        """
        print(X.shape[1])
        assert self.h < X.shape[1], '成分个数h不能大于样本的维度'  # 传进来的成分个数大于样本的维度时，抛出异常
        rfpls_model = RF_PLS(self.h, n_estimators=self.n_estimators, max_depth=self.max_depth)
        y_predict, y_RR, y_RMSE = rfpls_model.train(train_x, train_y)  # 训练
        print(y_RMSE)

        # 预测训练集
        y_predict, y_RR, y_RMSE = rfpls_model.predict(train_x, train_y)
        print("训练集", y_RMSE)

        # 预测测试集
        y_te_predict, y_te_RR, y_te_RMSE = rfpls_model.predict(test_x, test_y)
        print("测试集", y_te_RMSE)

        self.res_dict = {
            '训练集RMSE': y_RMSE,
            '测试集RMSE': y_te_RMSE
        }

    def getRes(self):
        return self.res_dict


if __name__ == '__main__':
    # 步骤1：读取数据
    # 1.1 blogData_test：1个因变量
    df_xy = pd.read_csv('../data/blogData_test1.csv')
    # print(df_xy)
    # print(df_xy.shape)
    xname_list = df_xy.columns.values.tolist()[0:df_xy.shape[1] - 1]
    #
    # X = df_xy[xname_list]
    # print(X.shape)
    # y = df_xy['y']
    #
    # # 步骤2：划分训练集测试集
    # train_x, train_y, test_x, test_y = splitDataSet(X.values, y.values.reshape(X.shape[0], 1), q=0.8)
    var_dict = {
        'independ_var': xname_list,
        'depend_var': ['y']
    }
    parameter_dict = {
        'q': 0.8,
        'h': 10,
        'n_estimators': 2,
        'max_depth': None
    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunRFPLS(df_xy, all_dict)
    r.run()
    # # 步骤:3：建模
    # """
    # RF_PLS(h, n_estimators=10, max_depth=None)
    # h = 10  # 成分个数，默认为10，其应小于等于样本的维度，从前台传进来应注意
    # n_estimators = 10  # 随机森林中的树的数量，默认为10，可自行设置
    # max_depth = None  # 树的最大深度， 默认None，节点被扩展，直到所有叶子都是纯的（可自行设置）
    # """
    # h = 10
    # print(X.shape[1])
    # assert h < X.shape[1], '成分个数h不能大于样本的维度'  # 传进来的成分个数大于样本的维度时，抛出异常
    # rfpls_model = RF_PLS(h, n_estimators=10, max_depth=None)
    # y_predict, y_RR, y_RMSE = rfpls_model.train(train_x, train_y)  # 训练
    # print(y_RMSE)
    #
    # # 预测训练集
    # y_predict, y_RR, y_RMSE = rfpls_model.predict(train_x, train_y)
    # print("训练集", y_RMSE)
    #
    # # 预测测试集
    # y_te_predict, y_te_RR, y_te_RMSE = rfpls_model.predict(test_x, test_y)
    # print("测试集", y_te_RMSE)



