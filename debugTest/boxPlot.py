#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/6/10 0010 10:26
#@Author  :    tb_youth
#@FileName:    boxPlot.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
绘制箱线图
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#设置随机种子
np.random.seed(2)
#5*4,0-1数据放入DataFrame
df = pd.DataFrame(np.random.rand(5,4),columns=['A','B','C','D'])
print(df)
df.boxplot()
plt.show()



'''
x：指定要绘制箱线图的数据；
notch：是否是凹口的形式展现箱线图，默认非凹口；
sym：指定异常点的形状，默认为+号显示；
vert：是否需要将箱线图垂直摆放，默认垂直摆放；
whis：指定上下须与上下四分位的距离，默认为1.5倍的四分位差；
positions：指定箱线图的位置，默认为[0,1,2…]；
widths：指定箱线图的宽度，默认为0.5；
patch_artist：是否填充箱体的颜色；
meanline：是否用线的形式表示均值，默认用点来表示；
showmeans：是否显示均值，默认不显示；
showcaps：是否显示箱线图顶端和末端的两条线，默认显示；
showbox：是否显示箱线图的箱体，默认显示；
showfliers：是否显示异常值，默认显示；
boxprops：设置箱体的属性，如边框色，填充色等；
labels：为箱线图添加标签，类似于图例的作用；
filerprops：设置异常值的属性，如异常点的形状、大小、填充色等；
medianprops：设置中位数的属性，如线的类型、粗细等；
meanprops：设置均值的属性，如点的大小、颜色等；
capprops：设置箱线图顶端和末端线条的属性，如颜色、粗细等；
whiskerprops：设置须的属性，如颜色、粗细、线的类型等；
 
'''

# 设置图形的显示风格

plt.style.use('ggplot')

# 设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

plt.rcParams['axes.unicode_minus'] = False


plt.boxplot(x=df,  # 指定绘图数据

            patch_artist=True,  # 要求用自定义颜色填充盒形图，默认白色填充

            showmeans=True,  # 以点的形式显示均值

            boxprops={'color': 'black', 'facecolor': '#9999ff'},  # 设置箱体属性，填充色和边框色

            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},  # 设置异常值属性，点的形状、填充色和边框色

            meanprops={'marker': '+', 'markerfacecolor': 'indianred'},  # 设置均值点的属性，点的形状、填充色

            medianprops={'linestyle': '--', 'color': 'orange'})  # 设置中位数线的属性，线的类型和颜色



# 去除箱线图的上边框与右边框的刻度标签

plt.tick_params(top='off', right='off')

# 显示图形

plt.show()

