#coding:utf-8
# ---导入库---
import matplotlib.pyplot as plt
#from numpy import *   #尽量避免这样全部导入！
from algorthms import PLS
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.optimize import fmin_slsqp
from sklearn.cross_decomposition import PLSRegression
from algorthms.SplitDataSet import SplitDataHelper


np.random.seed(0)

## 数据包络分析
class SBM(object):
    ## 初始化
    def __init__(self, inputs, outputs):
        # supplied data
        self.inputs = inputs
        self.outputs = outputs
        # parameters
        ## 决策单元数(样本量)n, 投入要素（自变量X）m，产出（因变量y）r
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]
        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)
        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas
    ## 效率值计算
    def __efficiency(self, unit):
        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)
        return (numerator / denominator)[unit]
    ## 目标值计算
    def __target(self, x, unit):
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)
        return numerator / denominator
    ## 约束条件
    def __constraints(self, x, unit):
        '''x的权重， 决策单元序列'''
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]  # unroll the weights
        constr = []  # init the constraint array
        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t * self.inputs[unit, input] - lhs
            constr.append(cons)
        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)
        # for each unit, range(n）
        for u in self.unit_:
            constr.append(lambdas[u])
        return np.array(constr)
    ## 优化
    def __optimize(self):
        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # rand init weights
            x0 = np.random.rand(d0) - 0.5
            ## 多元函数的最小化算法的接口。
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m + self.r)], x0[(self.m + self.r):]
            self.efficiency[unit] = self.__efficiency(unit)
    ## 训练
    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency

class SBM_PLS():
    '''
    1、SBM分析提取有效样本
    2、有效样本进行PLS分析
    3、打印PLS模型结构
    '''
    def __init__(self,  n_components, inputs, outputs ):
        '''
        :param n_components:
        :param inputs: 投入要素
        :param outputs:产出
        '''
        self.pls_model = PLSRegression(n_components=n_components)    ## sklearn 库的PLS回归
        self.sbm_model = SBM(inputs=inputs, outputs=outputs)

    def SBM_train(self):
        efficinetyList = self.sbm_model.fit()
        effSampleIndexList = []
        for i in range(len(efficinetyList)):
            if efficinetyList[i] >= 1:
                effSampleIndexList.append(i)
        # print("效率值：", efficinetyList)
        # print("有效投入：", effSampleIndexList)
        return effSampleIndexList

    def sampleCut(self, fileName):
        ## 获取有效样本下标列表
        effSampleIndexList = self.SBM_train()
        ## 预处理后的文件,有效样本
        effSample = self.inputsData.iloc[effSampleIndexList, :]
        #effSample.to_csv(fileName)
        return effSample
        # print("====" * 10)
        # print("原始样本:", self.inputsData)
        # print("有效样本:", effSample)
        # print("\n======= PLS ==========")
        # PLS.Main(fileName, delimiter=delimiter)
        # print("\n======= SBM_PLS =============")
        # PLS.Main(cutFile, delimiter=delimiter)

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
        ## 受限玻尔兹曼机RBM映射后X，Y
        # self.X_tr_tranformed = self.DSA_train(X_tr)  # X训练集训练、映射
        # self.X_te_tranformed = self.tranform(X_te)  # X测试集映射
        # self.y_tr_tranformed = self.DSA_train(y_tr)  # y训练集训练、映射
        # self.y_te_tranformed = self.tranform(y_te)  # 训练集映射
        ## -------------------
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
        if SST != 0 :
            RR = SSR / SST
        else:
            RR = np.inf
        RMSE = np.sqrt(SSE / row)
        return RR, RMSE

# 运行算法接口
class RunSBMPLS:
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


    def run(self):
        ## 初始化参数，提取投入因素X，产出y
        self.initParameter()
        yNum = len(self.dependent_var)
        inputsData = self.df.iloc[:, :-yNum].values
        outputsData = self.df.iloc[:, -yNum:].values
        # 建模, 获取有效样本下标列表
        sbm_pls_model = SBM_PLS(self.n_components, inputs=inputsData, outputs=outputsData)
        effSampleIndexList = sbm_pls_model.SBM_train()

        # 将有效样本集数据，划分训练集测试集
        self.effSample = self.df.iloc[effSampleIndexList, :]
        X = self.effSample[self.independent_var]
        y = self.effSample[self.dependent_var]

        X = np.mat(X, dtype=float)
        y = np.mat(y, dtype=float)
        # print(X)
        # print(y)
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X, y, q=self.q)
        y0_tr_predict, y0_tr_RMSE = sbm_pls_model.train(train_x, test_x, train_y, test_y)
        print("训练集", y0_tr_RMSE)

        # 预测 test_x
        y0_te_predict, y0_te_RMSE =sbm_pls_model.predict()

        print("测试集", y0_te_RMSE)
        print(sbm_pls_model.X_te_tranformed.shape)

        print('-' * 100)
        print(test_y)
        print('-' * 100)
        print(y0_te_predict)
        print('-' * 100)

        predict_test = pd.DataFrame()
        predict_test['预测值'] = y0_te_predict[0]
        predict_test['真实值'] = test_y[0]
        print(predict_test)
        show_data_dict = {
            '预测值和真实值': predict_test
        }
        self.res_dict = {
            '训练集RMSE': y0_tr_RMSE,
            '测试集RMSE': y0_te_RMSE,
            'show_data_dict':show_data_dict
        }


    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        return self.res_dict

if __name__ == '__main__':
    # 读取数据
    filename2 = "../data/TCMdata.csv"
    df2 = pd.read_csv(filename2)
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
    }
    # 设置参数页面
    all_dict2 = {
        'var_dict': var_dict2,
        'parameter_dict': parameter_dict
    }
    r2 = RunSBMPLS(df2, all_dict2)
    r2.run()
    print("运行结束！")
