#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/22 0022 18:40
#@Author  :    tb_youth
#@FileName:    DSAPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


# 初始化四个参数 W1 , W2 , B1 , B2. 其中W2=W1.T;
# 因此需要初始化的params = [W1, B1, B2]

##################################################
# ---对输入数据中利用伯努利分布进行数据的腐化操作    #
# 即在伯努利分布随机采样数据进行置零从而使数据腐化   #
##################################################
# ---前向传播---
# 输入层 -> 中间层
# sigmoid(np.dot(input , W1) + B1)

# 中间层 -> 输出层
# sigmoid(np.dot(hidden , W2) + B2)

# ---反向传播，随机梯度下降法---
# 计算代价函数cost
#######################################################################
# 交叉熵，L(x,z) = -np.sum(x * np.log(z) + (1-x)*np.log(1-z) , axis=1) -#
#######################################################################
# cost = np.mean(L(x,z))

# 利用梯度进行修改 params = params - learning_rate * gparam
'''
    #随机产生高斯分布
    import numpy as np
    mean = (1,2)
    cov = [[1,0],[0,2]]
    x = np.random.multivariate_normal(mean, cov,5000)
    np.mean(x[:,1])
    np.var(x[:,1])

    #随机产生伯努利分布
    y = np.random.binomial(1,0.5,(1000,1000))
'''
# ---导入库---
# import matplotlib.pyplot as plt
from numpy import *
# from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from algorthms.SplitDataSet import SplitDataHelper
random.seed(0)


# ---产生一个随机矩阵
def rand(a, b):  # 产生一个a*b的随机矩阵
    return 2 * np.random.random((a, b)) - 1


# ---使用s函数
def s(x, deriv=False):
    x = np.array(x, dtype=np.float64)
    if (deriv == True):
        return x * (1 - x)
    return 1.0 / (1 + np.exp(-x))


# ---定义一个降噪自编码器
class DSA:
    def __init__(self, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05):
        # ---定义网络的值---
        self.ny = ny
        self.iterations = iterations
        self.beta = beta
        self.eta = eta
        self.sp = sp

    def corrupt(self, inputs):
        # 随机产生一个伯努利分布
        binom = np.random.binomial(1, 0.5, inputs.shape)
        return inputs * binom
        # -------正向传播---------

    def forward(self, inputs, ni, no):  # ni, no输入输出层的神经元个数，等于inputs的特征个数
        # ---生成权重矩阵---
        self.w1 = rand(ni + 1, self.ny)
        self.w2 = rand(self.ny + 1, no)
        # 随机产生一个伯努利分布
        binom = np.random.binomial(1, 0.5, inputs.shape)
        # 将输入数据基于伯努利分布进行腐化
        inputs = inputs * binom
        # 阈值b
        b = np.ones((len(inputs), 1))
        # Step00 原始数据 -> 输入层
        self.ai = np.c_[np.array(inputs), b]
        # Step01 输入层 -> 隐含层
        self.ah = s(np.dot(self.ai, self.w1))
        self.ay = np.c_[s(np.dot(self.ai, self.w1)), b]
        # Step02 隐含层 -> 输出层
        self.ao = s(np.dot(self.ay, self.w2))

    # -----反向传播--------
    def backward(self, targets):
        # ---数据的样本数维度---
        mi = len(targets)
        # ---Step00 真实输出值---
        self.at = np.array(targets)
        # ---加入稀疏性规则项---
        rho = (1.0 / mi) * sum(self.ah, 0)
        # ---计算KL散度---
        Jsparse = sum(self.sp * np.log(self.sp / rho) + (1 - self.sp) * np.log((1 - self.sp) / (1 - rho)))
        # ---稀疏规则项的偏导---
        sterm = self.beta * (-self.sp / rho + (1 - self.sp) / (1 - rho))
        # ---Step01 计算输出层的梯度---
        o_deltas = s(self.ao, True) * (self.at - self.ao)
        # ---Step02 计算隐含层的梯度,这里是加入了稀疏导项---
        y_deltas = s(self.ay, True) * (np.dot(o_deltas, self.w2.T) \
                                       - np.c_[np.tile(sterm, (mi, 1)), np.zeros((mi, 1))])
        # ---Step03 更新权值---
        self.w2 = self.w2 + (self.eta * 1.0) * (np.dot(self.ay.T, o_deltas))
        self.w1 = self.w1 + (self.eta * 1.0) * (np.dot(self.ai.T, y_deltas[:, :-1]))
        # ---计算代价J(w,b)给予输出显示---
        Jcost = sum((0.5 / mi) * sum((self.at - self.ao) ** 2))
        return Jcost + self.beta * Jsparse

    # ----训练函数-----
    def train(self, patterns):  # patterns:数据集；iterations:迭代次数；beta:
        inputs = np.array(patterns)
        error = 0.0
        liust = list()
        # plt.figure()
        for k in range(self.iterations):
            self.forward(inputs, inputs.shape[1], inputs.shape[1])  # 后面两个参数是输入输出神经元的个数，等于inputs的特征个数
            error = self.backward(inputs)
            liust.append(error)
            if k % 100 == 0:
                print('-' * 50)
                print(error)
        xh = range(self.iterations)
        # plt.plot(xh, liust, 'r')

    # ---取出中间层---
    def tranform(self, iput):
        # 阈值b
        b = np.ones((len(iput), 1))
        # Step00 原始数据 -> 输入层
        ai = np.c_[np.array(iput), b]
        # Step01 输入层 -> 隐含层
        ay = s(np.dot(ai, self.w1))  # sigmoid激活函数，ay代表编译后的矩阵，特征个数等于隐含层神经元的个数
        return ay




class DSA_PLS:
    def __init__(self, n_components, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05):
        self.pls_model = PLSRegression(n_components=n_components)
        self.dsa_model = DSA(ny=ny, iterations=iterations, beta=beta, eta=eta, sp=sp)

    def DSA_train(self, input):
        self.dsa_model.train(input)  # 训练
        input_transformed = self.dsa_model.tranform(input)
        return input_transformed

    def tranform(self, input):
        return self.dsa_model.tranform(input)

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
        # 受限玻尔兹曼机RBM映射后X，Y
        self.X_tr_tranformed = self.DSA_train(X_tr)  # X训练集训练、映射
        self.X_te_tranformed = self.tranform(X_te)  # X测试集映射
        self.y_tr_tranformed = y_tr  # y不进行映射
        self.y_te_tranformed = y_te  # y不进行映射
        # self.y_tr_tranformed = self.DSA_train(y_tr)  # y训练集训练、映射
        # self.y_te_tranformed = self.tranform(y_te)  # 训练集映射
        # PLS回归
        y0_tr_predict, y0_tr_RR, y0_tr_RMSE = self.PLS_train(self.X_tr_tranformed, self.y_tr_tranformed)
        return y0_tr_predict, y0_tr_RMSE

    def predict(self):  # 只能预测测试集
        y0_te_predict, y0_te_RR, y0_te_RMSE = self.PLS_predict(self.X_te_tranformed, self.y_te_tranformed)
        return y0_te_predict, y0_te_RMSE

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


# 运行算法接口
class RunDSAPLS:
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
        self.ny = parameter_dict.get('ny')
        self.iterations = parameter_dict.get('iterations')
        self.beta = parameter_dict.get('beta')
        self.eta = parameter_dict.get('eta')
        self.sp = parameter_dict.get('sp')

    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]
        X = np.mat(X,dtype=float)
        y = np.mat(y,dtype=float)
        # print(X)
        # print(y)

        # 划分训练集测试集
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X, y, q=self.q)

        # 建模
        dsa_pls_model = DSA_PLS(self.n_components, ny=self.ny, iterations=self.iterations,
                                beta=self.beta, eta=self.eta, sp=self.sp)
        y0_tr_predict, y0_tr_RMSE = dsa_pls_model.train(train_x, test_x, train_y, test_y)
        print("训练集", y0_tr_RMSE)
        y0_te_true = dsa_pls_model.y_te_tranformed  # 虽然还叫y_te_tranformed，但并未进行DSA映射，和test_y一致
        # 预测 test_x
        """
        修改了算法，y不进行DSA映射，若单因变量建模，y只有1维；多因变量建模，y有多维
        """
        y0_te_predict, y0_te_RMSE = dsa_pls_model.predict() # y预测值
        print("测试集", y0_te_RMSE)

        print(y0_te_true)
        print('-'*100)
        print(y0_te_predict)
        print('-'*100)
        # 单因变量建模，一个DataFrame显示就足够
        if len(self.dependent_var) == 1:
            predict_test = pd.DataFrame()
            dependent_str = str(self.dependent_var[0])
            predict_test['{}_预测值'.format(dependent_str)] = np.ravel(y0_te_predict)
            predict_test['{}_真实值'.format(dependent_str)] = np.ravel(test_y)
            show_data_dict = {
                '预测值和真实值': predict_test
            }
        else:
            # 预测值和真实值，若是多因变量建模，需要两个DataFrame来显示
            predict_test_predict = pd.DataFrame(y0_te_predict)
            predict_test_predict.columns = self.dependent_var
            predict_test_true = pd.DataFrame(y0_te_true)
            predict_test_true.columns = self.dependent_var
            show_data_dict = {
                '预测值': predict_test_predict,
                '真实值': predict_test_true
            }

        self.res_dict = {
            '训练集RMSE':y0_tr_RMSE,
            '测试集RMSE':y0_te_RMSE,
            'show_data_dict':show_data_dict
        }

        print(dsa_pls_model.X_te_tranformed.shape)

    # 获取结果中需要用到的数据（展示或画图所用数据）接口
    def getRes(self):
        return self.res_dict


# ---实例---
if __name__ == '__main__':
    """
    (ni, ny, no)
    ni:输入层神经元个数，等于数据矩阵的特征个数
    ny：隐含层神经元个数，一般大于输入层的神经元个数，默认22，可自行设置
    no：输出层神经元个数，等于数据矩阵的特征个数
    """

    """
    train(self, X_tr, X_te, y_tr, y_te, h, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    X_tr, X_te, y_tr, y_te:数据矩阵
    h:成分个数，可自行设置，但是不能大于输入矩阵的特征个数
    ny：隐含层神经元个数，一般大于输入层的神经元个数，默认22，可自行设置
    iterations：迭代次数，默认为1000，可自行设置
    beta：代价函数中的一个压缩参数，默认0.5，可自行设置
    eta：步长，相当于学习率，默认为1，可自行设置
    sp：稀疏性参数，通常是一个接近于0的较小值，默认为0.05，可自行设置
    """

    # 读取数据
    df = pd.read_excel("../data/DSAPLS_test.xlsx")
    print(df.shape)

    # 变量字典：自变量，因变量
    var_dict = {
        'independ_var': ["x1", "x2", "x3", "x4"],
        'depend_var': ["y"]
    }
    # 参数字典，需要设置的参数
    parameter_dict = {
        'q': 0.7,
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
    r = RunDSAPLS(df, all_dict)
    r.run()
    for key,value in r.getRes().items():
        print(key)
        print(value)

