#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/3 0003 0:09
#@Author  :    tb_youth
#@FileName:    scatter0.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
n=1000
#rand 均匀分布和 randn高斯分布
x=np.random.randn(1,n)
y=np.random.randn(1,n)
T=np.arctan2(x,y)
plt.scatter(x,y,c=T,s=25,alpha=0.4,marker='o')
#T:散点的颜色
#s：散点的大小
#alpha:是透明程度
plt.show()
