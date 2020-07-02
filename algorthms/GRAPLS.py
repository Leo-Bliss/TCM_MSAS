#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/22 0022 18:40
#@Author  :    tb_youth
#@FileName:    GRAAPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from algorthms.SplitDataSet import SplitDataHelper


np.random.seed(0)
## 灰度关联分析
class GRA(object):
    ## 初始化
    def __init__(self, inputs):
        # supplied data
        self.inputs = inputs
        self.row,self.col = inputs.shape

    ## 计算第 m 列与其他列的关联系数, 分辨系数:bata
    def getGRA_ONE(self, col=0, beta=0.5):
        '''
        :param DataFrame: 输入数据 df格式
        :param m: 参考数列 index
        :return:
        '''
        ## 数据标准化
        self.inputs = (self.inputs - self.inputs.min()) / (self.inputs.max() - self.inputs.min())
        ## 设置比较数列，参考数列
        ce = self.inputs.iloc[:, 0:]
        std = self.inputs.iloc[:, col]
        ## 参考数列的样本大小， 列数量
        n,m = ce.shape
        # 与标准要素比较，相减
        aList = np.zeros([m, n])
        # ce = pd.DataFrame(ce,dtype=float)
        # std = pd.Series(std,dtype=float)
        for i in range(m):
            for j in range(n):
                aList[i, j] = abs(ce.iloc[j, i] - std[j])
        # 取出矩阵中最大值与最小值
        maxValue = np.amax(aList)
        minValue = np.amin(aList)
        # 计算值
        result = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                result[i, j] = (minValue + beta * maxValue) / (aList[i, j] + beta * maxValue)
        # 求均值，得到灰色关联值
        result2 = np.zeros(m)
        for i in range(m):
            result2[i] = np.mean(result[i, :])
        RT = pd.DataFrame(result2)
        return RT

    ## 计算任意两列直接的 person系数
    def getGRA(self, beta=0.5):
        list_columns = [str(s) for s in range(len(self.inputs.columns)) if s not in [None]]
        graDf = pd.DataFrame(columns=list_columns,dtype=float)
        length = len(self.inputs.columns)
        for i in range(length):
            graDf.iloc[:, i] = self.getGRA_ONE(col=i, beta=beta)[0]
            # print(df_local.iloc[:,i])
        return graDf

    ## 确定因变量中参考列下标
    def getMaxCorrIndex(self):
        ## 数据各变量的相关系数矩阵
        corrDf = self.inputs.corr().abs()
        ## 矩阵的大小
        corrSize = corrDf.shape[0]
        ## 查找相关系数值最大的列 Xi
        maxValue, maxCol = 0, 0
        for i in range(corrSize):
            for j in range(0, i):
                if corrDf.iloc[i, j] > maxValue:
                    maxValue = corrDf.iloc[i, j]
                    maxCol = j
        return maxValue, maxCol

    ## 这里的绘图会影响界面上的绘图！！！！

    #灰色关联结果矩阵可视化
    # def ShowGRAHeatMap(self, graDf):
    #     # %matplotlib inline
    #     colormap = plt.cm.RdBu
    #     # plt.figure(figsize=(14, 12))
    #     plt.title('Pearson Correlation of Features', y=1.05, size=15)
    #     sns.heatmap(graDf.astype(float), linewidths=0.1, vmax=1.0, square=True,
    #                 cmap=colormap, linecolor='white', annot=True)
    #     plt.show()

    ## 保存
    def saveGRA(self, fileName, graDf):
        graDf.to_csv(fileName)

class GRA_PLS():
    '''
    1、SBM分析提取有效样本
    2、有效样本进行PLS分析
    3、打印PLS模型结构
    '''
    def __init__(self,  n_components, inputs):
        '''
        :param n_components:
        :param inputs: 投入要素
        :param outputs:产出
        '''
        self.inputs = inputs
        self.pls_model = PLSRegression(n_components=n_components)    ## sklearn 库的PLS回归
        self.gra_model = GRA(inputs = self.inputs)

    def GRA_train(self, beta=0.5):
        '''
        :return: 自变量中剔除变量的下标
        '''
        ## 确定参考数列Index
        maxValue, maxCol = self.gra_model.getMaxCorrIndex()
        graDf = self.gra_model.getGRA(beta=beta)
        maxColGra = graDf.iloc[maxCol].tolist()
        ## 最大相关系数中的最小灰色关联度变量下标
        minGraIndex = maxColGra.index(min(maxColGra))
        print("GRA系数矩阵：", graDf)
        print("参考数列:{}； 其GRA系数:{}； 剔除列下标Index：{}".format(maxCol, maxColGra, minGraIndex))
        ## 关联系数可视化
        # self.gra_model.ShowGRAHeatMap(graDf=graDf)
        return maxCol, minGraIndex

    def sampleCut(self):
        ## 获取有效样本下标列表
        maxCol, minGraIndex = self.GRA_train()
        ## 预处理后的文件,有效样本
        effSample = self.inputs.drop(self.inputs.columns[minGraIndex], axis=1)
        return effSample
        # print("====" * 10)
        # print("原始样本:", self.inputs)
        # print("有效样本:", effSample)

    def PLS_train(self, x0_tr, y0_tr):
        # 训练模型，预测训练集
        self.pls_model.fit(x0_tr, y0_tr)
        # 取出一些结果，比如成分个数，回归系数这些
        self.coefficient = self.pls_model.coef_
        y0_tr_predict = self.pls_model.predict(x0_tr)
        y0_tr_RR, y0_tr_RMSE = self.getRRandRMSE(y0_tr, y0_tr_predict)
        # print("y0_tr_RR", y0_tr_RR)
        # y1_tr_RR = self.pls_model.score(x0_tr, y0_tr)
        # print("y1_tr_RR", y1_tr_RR)
        return y0_tr_predict, y0_tr_RR, y0_tr_RMSE

    def PLS_predict(self, x0_te, y0_te):  # 其实训练集可测试集都可以用
        # 预测测试集，!!! 不行，如果没分训练集和测试集呢(还没想好怎么写),这里应该是这样写
        y0_te_predict = self.pls_model.predict(x0_te)
        y0_te_RR, y0_te_RMSE = self.getRRandRMSE(y0_te, y0_te_predict)
        return y0_te_predict, y0_te_RR, y0_te_RMSE

    def train(self, X_tr, X_te, y_tr, y_te):
        self.X_tr_tranformed = X_tr
        self.X_te_tranformed = X_te
        self.y_tr_tranformed = y_tr
        self.y_te_tranformed = y_te
        # PLS回归
        y0_tr_predict, y0_tr_RR, y0_tr_RMSE = self.PLS_train(self.X_tr_tranformed, self.y_tr_tranformed)
        return y0_tr_predict, y0_tr_RMSE

    def predict(self):  # 只能预测测试集
        y0_te_predict, y0_te_RR, y0_te_RMSE = self.PLS_predict(self.X_te_tranformed, self.y_te_tranformed)
        return y0_te_predict, y0_te_RMSE

    def getRRandRMSE(self, y0, y0_predict):
        row = np.shape(y0)[0]
        mean_y = np.mean(y0, 0)
        y_mean = np.tile(mean_y, (row, 1))
        SSE = sum(sum(np.power((y0 - y0_predict), 2), 0))
        SST = sum(sum(np.power((y0 - y_mean), 2), 0))
        SSR = sum(sum(np.power((y0_predict - y_mean), 2), 0))
        if SST:
            RR = SSR/SST
        else:
            RR = np.inf
            print("waring:SST=0!")
        RMSE = np.sqrt(SSE/row)
        return RR, RMSE

# 运行算法接口
class RunGRAPLS:
    def __init__(self, df, all_dict):
        ## dataFrame样本数据集， 相关参数字典
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
        self.beta = parameter_dict.get('beta')

    def run(self):
        ## 初始化参数，自变量X，因变量y
        self.initParameter()
        inputsData = self.df[self.independent_var]
        inputsData.dtype = float

        # 建模, 获取有效样本下标列表
        gra_pls_model = GRA_PLS(self.n_components, inputs=inputsData)
        maxCol, minGraIndex = gra_pls_model.GRA_train(self.beta)
        ## 提取弱相关的列 minGraIndex
        del self.independent_var[minGraIndex]

        # 将有效样本集数据，划分训练集测试集
        self.effSample = self.df.drop(self.df.columns[minGraIndex], axis=1)
        X = self.effSample[self.independent_var]
        y = self.effSample[self.dependent_var]
        # print("====" * 10)
        # print("原始样本:", self.df)
        # print("有效样本:", self.effSample)

        X = np.mat(X, dtype=float)
        y = np.mat(y, dtype=float)
        # print(X)
        # print(y)
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X, y, q=self.q)
        y0_tr_predict, y0_tr_RMSE = gra_pls_model.train(train_x, test_x, train_y, test_y)
        print("训练集", y0_tr_RMSE)
        # 预测 test_x
        y0_te_predict, y0_te_RMSE = gra_pls_model.predict()

        print("测试集", y0_te_RMSE)

        print(test_y)
        print('-' * 100)
        print(y0_te_predict)
        print('-' * 100)
        dependent_str = str(self.dependent_var[0])
        predict_test = pd.DataFrame()
        predict_test['{}_预测值'.format(dependent_str)] = np.ravel(y0_te_predict)
        predict_test['{}_真实值'.format(dependent_str)] = np.ravel(test_y)
        print(predict_test)

        show_data_dict = {
            '预测值和真实值': predict_test
        }

        self.res_dict = {
            '训练集RMSE': y0_tr_RMSE[0,0],
            '测试集RMSE': y0_te_RMSE[0,0],
            'show_data_dict':show_data_dict
        }

        print(gra_pls_model.X_te_tranformed)

    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        return self.res_dict


if __name__ == '__main__':

    # 读取数据
    filename2 = "../data/GRAPLS_test.csv"
    df2 = pd.read_csv(filename2)
    print(df2.shape)

    # 变量字典：自变量，因变量
    var_dict = {
        'independ_var': ["x1", "x2", "x3", "x4"],
        'depend_var': ["y"]
    }

    var_dict2 = {
        'independ_var': ["x1", "x2", "x3", "x4","x5", "x6", "x7", "x8","x9"],
        'depend_var': ["y1"]
    }
    # 参数字典，需要设置的参数
    parameter_dict = {
        'q': 0.7,
        'n_components': 2,
        'beta': 0.5,
        'eta': 1
    }
    # 设置参数页面
    all_dict2 = {
        'var_dict': var_dict2,
        'parameter_dict': parameter_dict
    }
    r2 = RunGRAPLS(df2, all_dict2)
    r2.run()
    for key,value in r2.getRes().items():
        print(key)
        print(value)

    print("运行结束！")
