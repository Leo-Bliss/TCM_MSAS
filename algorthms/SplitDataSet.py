#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/22 0022 18:30
#@Author  :    tb_youth
#@FileName:    SplitDataSet.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

from numpy import *
import random  # 这个包要在下面，不然会被numpy中的random方法覆盖，会报错
random.seed(0)

class SplitDataHelper:
    def __init__(self):
        pass

    def splitDataSet(self,x, y, q=0.7):  # q表示训练集的样本占比, x,y不能是DataFrame类型
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