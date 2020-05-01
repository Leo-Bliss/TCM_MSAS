#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/3 0003 22:00
#@Author  :    tb_youth
#@FileName:    EnumAlgorithms.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
用枚举类来定义各个算法对应的编号,
如果不使用官方的包的话，可以自己定义字典或者一个专门的类
'''
#暂时还没有将枚举类应用到本项目中,估计后期维护会用到

from enum import Enum,unique

@unique #成员值唯一
class Algorithms(Enum):
    DSA_PLS = 1
    SBMPLS = 2
    PLSCF = 3
    LAPLS = 4
    GRA_PLS = 5
    RBM_PLS = 6
    SEA_PLS = 7
    DBN_PLS = 8
    Mtree_PLS = 9
    RF_PLS = 10
    PLS_S_DA = 11

if __name__ == '__main__':
    print(type(Algorithms.LAPLS),Algorithms.LAPLS.name)
    print(Algorithms.LAPLS.value)
    print(repr(Algorithms.LAPLS))
    # for al in Algorithms:
    #     print(al)

