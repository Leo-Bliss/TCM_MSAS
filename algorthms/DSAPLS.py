# -*- coding: utf-8 -*-

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
import matplotlib.pyplot as plt
from numpy import * #尽量避免这样全部导入！
from sklearn import preprocessing
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


class PLS:
    def __init__(self, x0, y0):
        self.r = corrcoef(x0)
        self.m = shape(x0)[1]
        self.n = shape(y0)[1]  # 自变量和因变量个数
        self.row = shape(x0)[0]
        # self.n_components = 0  # 存放最终的成分个数 这些可以不用初始化空间
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
        self.n_components, self.W_star, self.T, beta = self.Pls(x0, y0, h)

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

    def predict(self, x0_te):  # 这个是针对测试集，训练集也可以用
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
        self.y_tr_tranformed = self.DSA_train(y_tr)  # y训练集训练、映射
        self.y_te_tranformed = self.tranform(y_te)  # 训练集映射
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

        # 预测 test_x
        y0_te_predict, y0_te_RMSE = dsa_pls_model.predict()
        print("测试集", y0_te_RMSE)
        self.res_dict = {
            '训练集RMSE':y0_tr_RMSE,
            '测试集RMSE':y0_te_RMSE
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
    df = pd.read_excel("../data/data01.xlsx")

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

    # X = self.df[["x1", "x2", "x3", "x4"]]
    # y = self.df[['y']]
    #
    # X = np.mat(X)
    # y = np.mat(y)
    # print(X)
    # print(y)
    #
    # # 划分训练集测试集
    # train_x, train_y, test_x, test_y = splitDataSet(X, y, q=0.7)
    #
    # # 建模
    # dsa_pls_model = DSA_PLS(2, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    # y0_tr_predict, y0_tr_RMSE = dsa_pls_model.train(train_x, test_x, train_y, test_y)
    # print("训练集", y0_tr_RMSE)
    #
    # # 预测 test_x
    # y0_te_predict, y0_te_RMSE = dsa_pls_model.predict()
    # print("测试集", y0_te_RMSE)
    #
    # print(dsa_pls_model.X_te_tranformed.shape)

    # pat = [[0,0, 0,0],
    #     [0,1, 0,1],
    #     [1,0, 1,0],
    #     [1,1, 1,1]]
    # n = DSA_PLS(2, ny=22, iterations=1000, beta=0.5, eta=1, sp=0.05)
    # ab = n.DSA_train(np.array(pat))
    # ah = n.tranform(np.array(pat))
    # print(ab)
    # print(ah)
    # print(ah.shape)

    # pat = [[0, 0, 0, 0],
    #        [0, 1, 0, 1],
    #        [1, 0, 1, 0],
    #        [1, 1, 1, 1]]
    # n = DSA(4, 22, 4)
    # n.train(np.array(pat), 1000, 0.5, 1, 0.05)  #train(self, patterns, iterations, beta, eta, sp):  # patterns:数据集；iterations:迭代次数；beta:；；sp:
    # ah = n.tranform(np.array(pat))
    # print(ah.shape)
    # print(np.zeros((3, 2)))
