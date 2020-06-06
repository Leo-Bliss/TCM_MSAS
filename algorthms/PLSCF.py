#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/21 0021 21:02
#@Author  :    tb_youth
#@FileName:    PLSCF.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class CFS(object):
    def __init__(self):  # 这里不需要传什么进来，应该在这里把列标弄好
        super(CFS, self).__init__()

    def getFirstBestMerit(self, X_df, y):
        """
        获取X中与y相关系数最高的特征
        :param X_df: 要求是Dataframe类型的
        :param y: 要求是一行或者一列
        :return: 得分最大的特征的分数以及特征
        """
        # 当k = 1时，Merit得分相当于特征f和y的相关系数得分
        # 获取特征子集的特征名称
        x_name = X_df.columns.values.tolist()
        Corrcoefs = pd.Series()  # 存放每个特征的得分
        for i in x_name:
            Corrcoefs[i] = np.abs(np.corrcoef(y, X_df[i])[0][1])  # np.corrcoef(y, X_df[i])的结果是2*2的矩阵
        # 对分数降序排序取最大值以及对应的索引
        Corrcoefs = Corrcoefs.sort_values(ascending=False)
        return Corrcoefs[0], Corrcoefs.index.tolist()[0]  # 分数、特征

    # 单个备选特征f在评分函数下的得分
    def getMeritFor_f(self, X_selected, f, y):
        """单个备选特征f在评分函数下的得分
        CFS的评分函数：Merit

        :param X_selected: 已选特征子集,要求是Dataframe的类型
        :param f: 备选特征, 要求是一行或者一列
        :param y: label特征
        :return: 返回f在评分函数下的所得值
        """
        y = pd.Series(y)
        # 获取已选特征子集的特征名称
        x_selected_name = X_selected.columns.values.tolist()

        k = len(x_selected_name) + 1  # k表示当前特征子集S的特征个数
        r_cf_list = []  # 存放相关性，r_cf表示y与当前特征子集S(f + X_selected)的平均相关系数
        r_ff_list = []  # 存放冗余性，r_ff表示已选特征子集(X_selected)与特征f的平均相关系数
        r_cf_i = np.abs(np.corrcoef(y, f)[0][1])  # Pearson相关系数
        r_cf_list.append(r_cf_i)
        for i in x_selected_name:
            # 计算Pearson相关系数（在X_selected[i]下f和y的相关性）
            r_cf_i = np.abs(np.corrcoef(y, X_selected[i])[0][1])  # 取绝对值为相关性，绝对值越大，相关性越强
            r_cf_list.append(r_cf_i)
            r_ff_i = np.abs(np.corrcoef(X_selected[i], f)[0][1])  # 取绝对值为冗余性
            r_ff_list.append(r_ff_i)

        Merit = k * np.mean(r_cf_list) / np.sqrt(k + k * (k - 1) * np.mean(r_ff_list))
        return Merit

    # 备选特征子集在评分函数下的得分
    def getBestMerit(self, X_selected, X_unselected, y):
        """备选特征子集中得分最高的特征
        :param X_selected: 已选特征子集（Dataframe类型）
        :param X_unselected: 未选特征子集（Dataframe类型）
        :param y: label特征
        :return: 根据评分函数返回未选特征子集得分最高的特征
        """
        # # 获取已选特征子集的特征名称
        # x_selected_name = X_selected.columns.values.tolist()
        # 获取未选特征的特征名称
        x_unselected_name = X_unselected.columns.values.tolist()
        Merits = pd.Series()  # 存放每个未选特征的得分
        for i in x_unselected_name:
            Merits[i] = self.getMeritFor_f(X_selected, X_unselected[i], y)
        # 对分数降序排序取最大值以及对应的索引
        Merits = Merits.sort_values(ascending=False)
        return Merits[0], Merits.index.tolist()[0]  # 分数、特征  该评分函数下最好的特征


    def train(self, X_df, y):
        """
        CFS进行特征选择
        :param X_df: 训练集，Dataframe类型
        :param y: label特征
        :return:
        """
        X_selected = pd.DataFrame()  # 存放已选特征: 初始状态下已选特征为空
        X_unselected = pd.DataFrame(X_df)  # 存放未选特征: 初始状态下整个特征子集都未选择

        # 1.根据Merit选取第一个特征
        Merit_value, x_name_of_Merit_value = self.getFirstBestMerit(X_df, y)
        X_selected[x_name_of_Merit_value] = X_df[x_name_of_Merit_value]  # X_selected加入
        X_unselected = X_unselected.drop([x_name_of_Merit_value], axis=1)  # X_unselected删除
        before_Merit = Merit_value

        # 2.评分函数Merit加前向搜索算法进行特征选择
        while not (X_unselected.empty):
            # 找到评分最大的特征, 加入到已选特征中
            best_Merit, x_name_of_BestMerit = self.getBestMerit(X_selected, X_unselected, y)
            # 如果这一步最好的Merit小于等于上一步的，证明新加入的特征没有起到积极作用
            if best_Merit <= before_Merit:
                # 实际上，出现这种情况之后，后面所有的特征都不会有更好的了，因为best_Merit已经是最好的结果
                # 所以可以直接退出循环
                # X_unselected = X_unselected.drop([x_name_of_BestMerit], axis=1)  # X_unselected删除
                # continue
                break
            before_Merit = best_Merit
            X_selected[x_name_of_BestMerit] = X_df[x_name_of_BestMerit]  # X_selected加入
            X_unselected = X_unselected.drop([x_name_of_BestMerit], axis=1)  # X_unselected删除

        self.X_selected_name = X_selected.columns.values.tolist()

        # 3 返回已选特征子集
        return self.X_selected_name, before_Merit  # 返回已选特征子集，以及该特征子集对应的最好的Merit值

    def predict(self, X_df):
        """
        预测，也即获取特征选择之后的特征子集
        :param X_df: Dataframe类型
        :return:
        """
        return X_df[self.X_selected_name]

class PLSCF:
    def __init__(self, n_components):
        self.cfs_model = CFS()
        self.pls_model = PLSRegression(n_components=n_components)

    def getRRandRMSE(self, y0, y0_predict):
        RR = r2_score(y0, y0_predict)
        RMSE = np.sqrt(mean_squared_error(y0, y0_predict))
        return RR, RMSE

    def train(self, X_df, y):
        # 使用CFS进行特征选择
        self.X_selected_name, Merit = self.cfs_model.train(X_df, y)
        X_selected = X_df[self.X_selected_name]
        # PLS训练模型
        self.pls_model.fit(X_selected.values, y)
        # 取出一些结果，比如成分个数，回归系数这些
        self.coefficient = self.pls_model.coef_

        y_tr_predict = self.pls_model.predict(X_selected.values)
        y_tr_RR, y_tr_RMSE = self.getRRandRMSE(y, y_tr_predict)
        return y_tr_predict, y_tr_RR, y_tr_RMSE

    def predict(self, X_df, y):
        X_selected = X_df[self.X_selected_name]
        y_predict = self.pls_model.predict(X_selected.values)
        y_RR, y_RMSE = self.getRRandRMSE(y, y_predict)
        return y_predict, y_RR, y_RMSE

    def getSubX(self, X_df):
        return X_df[self.X_selected_name]

def getXnameList(column_Num):
    """
    为每个特征分配列名
    :param column_Num: 样本的维度，也即列的数量
    :return:
    """
    xname_list = []
    for i in range(1, column_Num+1):
        xname_i = "x" + str(i)
        xname_list.append(xname_i)
    return xname_list

class RunPLSCF:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict
        self.res_dict = {}

    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.train_size = parameter_dict.get('q')


    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        # 当因变量，这里不能设置为：self.df[self.dependent_var]!!!
        y = self.df[self.dependent_var[0]]
        X = pd.DataFrame(X,dtype=float)
        y = pd.Series(y,dtype=float)
        # 步骤2：划分训练集测试集
        train_x, test_x, train_y, test_y = train_test_split(X.values, y.values, test_size=self.train_size, random_state=0)

        # 训练集和测试集必须含有列标，而且列标要是字符串类型； 如果没有列标或者列标是整形，可以调用getXnameList()方法
        # xname_list_demo = getXnameList(X.shape[1])
        train_x = pd.DataFrame(train_x, columns=self.independent_var)
        test_x = pd.DataFrame(test_x, columns=self.independent_var)

        # 步骤:3：建模
        print('-'*100)
        """
        PLSCF(n_components=1)
        n_components：成分个数，默认1，不需要从前台传，因为特征选择之后不知道有多少特征，很容易出问题
        """
        plscf_model = PLSCF(n_components=1)
        y1_predict, y_RR, y_RMSE = plscf_model.train(train_x, train_y)  # 训练

        print(y_RMSE)

        # 预测测试集
        y_te_predict, y_te_RR, y_te_RMSE = plscf_model.predict(test_x, test_y)
        print("测试集", y_te_RMSE)

        print('-' * 100)
        print(test_y)
        print('-' * 100)
        print(y_te_predict)
        print('-' * 100)
        predict_test = pd.DataFrame()
        predict_test['预测值'] = pd.DataFrame(y_te_predict)[0]
        predict_test['真实值'] = pd.DataFrame(test_y)[0]
        print(predict_test)

        show_data_dict = {
            '预测值和真实值': predict_test
        }
        self.res_dict = {
            '训练集RMSE': y_RMSE,
            '测试集RMSE': y_te_RMSE,
            'show_data_dict':show_data_dict
        }

        # 获取特征子集
        sub_X = plscf_model.getSubX(X)

    def getRes(self):
        return self.res_dict


if __name__ == '__main__':
    # 步骤1：读取数据
    # 1.1 TCMdata：1个因变量
    df_xy = pd.read_excel('../data/TCMdata.xlsx',index_col=0)
    # print(df_xy)
    # print(df_xy.shape)
    xname_list = df_xy.columns.values.tolist()[0:df_xy.shape[1] - 1]

    # X = df_xy[xname_list]
    # # print(X.shape)
    # y = df_xy['y']
    var_dict = {
        'independ_var': xname_list,
        'depend_var': ['y']
    }
    parameter_dict = {
        'n_components': 1,
    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunPLSCF(df_xy, all_dict)
    r.run()
