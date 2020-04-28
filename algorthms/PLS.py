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

from numpy import *   #尽量避免这样全部导入！
from sklearn import preprocessing
import numpy as np



random.seed(0)
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
        ## 相关变量参数
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
        ## 成分提取
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
            ## 逐个样本计算PRESS
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

if __name__ == '__main__':
    pass