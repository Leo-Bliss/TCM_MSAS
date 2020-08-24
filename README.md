# 微型统计分析系统
# MSAS

>Mini Statistical Analysis System

### 项目地址

github: [https://github.com/tbyouth/TCM_MSAS](https://github.com/tbyouth/TCM_MSAS)

### 项目预览
[![dr35Wj.png](https://s1.ax1x.com/2020/08/24/dr35Wj.png)](https://imgchr.com/i/dr35Wj)

![dr3oSs.png](https://s1.ax1x.com/2020/08/24/dr3oSs.png)

[![dr3HO0.png](https://s1.ax1x.com/2020/08/24/dr3HO0.png)](https://imgchr.com/i/dr3HO0)

### 简介
本项目是基于PyQt框架开发的一个微型的数据统计分析系统。

该项目目前已经完成了版本v1.0的开发工作,

并且以**偏最小二乘的多功能统计分析系统**v1.0版本打包发布（软件著作权归江西中医药大学计算机学院所有），

该发布版本有幸选为江西中医药大学杜建强教授的论著《偏最小二乘法优化及其在中医药领域的应用研究》的配套软件。

本项目后续的的功能开发和算法集成工作正在进行中...

### 功能:
* 支持excel表格数据**基本的**预处理：

    + 数据增删改;
    
    + 数据剪切复制粘贴；
    
    + 数据检索高亮显示;
    
    + 数据查找替换；
    
    + 数据的导入导出等功能；

* 支持优化的偏最小二乘(PLS)数据分析功能,集成的PLS相关算法有11个:
    + 数据预处理模型2个：
    
        DSA-PLS：Denoising Sparse Autoencoder（降噪稀疏自编码器）融合降噪稀疏自编码器的偏最小二乘算法
        
        SBMPLS：Slacks Based Measure（非径向 DEA 模型）融合非径向数据包络分析的偏最小二乘算法
        
    + 特征选择模型3个：
    
        PLSCF：PLS feature selection based on feature correlation 基于特征相关的偏最小二乘特征选择算法
        
        LAPLS：Feature Selection Method Based on Patial Least Squares 基于偏最小二乘的特征选择算法
        
        GRA-PLS：Grey Relation Analysis-Partial Least Square 灰色关联的偏最小二乘辅助分析算法
        
    + 非线性特征提取模型3个：
    
        RBM-PLS：Restricted Boltzmann Machine（受限玻尔兹曼机）融合受限玻尔兹曼机的偏最小二乘优化模型
        
        SEA-PLS：融合稀疏自编码器的偏最小二乘优化模型
        
        DBN-PLS：Deep Belief Nets（深度置信网络）融合深度置信网络的偏最小二乘优化模型
    
    + 非线性回归模型3个：
   
        Mtree-PLS：Partial least squares method based on fusion model tree 融合模型树的偏最小二乘算法
        
        RF-PLS：PLS method for fusion of random forests 融合随机森林的偏最小二乘算法
        
        PLS-S-DA：Partial least squares discriminant analysis based on softmax 融合 softmax 的偏最小二乘判别分析算法
    
   

* 支持多线程运行算法模型

* 支持可视化分析功能：
     
    绘制可视化图形，图形的移动，放缩查看，图形的导出；

    由于时间仓促，学业繁忙（准备考研ing）等原因目前版本只集成了折线图，散点图，柱状图这三种可视化模型，
    
    更多可视化分析模型将在后续的开发中加入。
    
    

### 安装

1. 直接下载本项目zip文件源码或使用git下载项目源码

    ```shell
    $ git clone git@github.com:tbyouth/TCM_MSAS.git
    ```
2. 安装好python并配置好环境，打开命令行工具，

    使用 `pip install -r requirements.txt` 安装相关依赖模块；
    
    如果您使用pip安装依赖模块市下载太慢，建议您使用：
    
    `pip install -i https://pypi.douban.com/simple  -r requirements.txt`
    
    调用python 执行项目目录下的**runApp.py**:
    
	```py
	python runApp.py
	```

3. 此外，你可以[下载本项目打包发布的Release v1.0](https://github.com/tbyouth/TCM_MSAS/releases/tag/v1.0)使用，也可以使用pyinstaller进行编译打包成exe后使用，

    编译打包成exe语句：pyinstaller -i logo.ico -F src/MainWindow.py


#### 功能TODO
- [ ] 软件皮肤设置.
- [ ] 数据库导入导出数据.
- [ ] 数据预处理操作撤销回退.
- [ ] 数据可视化模框重构.
- [ ] 集成更多数据分析算法.
- [ ] 集成更多可视化分析模型.
- [ ] 软件在线更新.
- [ ] 代码优化...

#### 开发TODO
- [ ] 加入日志方便调试.

#### 说明
本项目由我独立开发，所以部分界面的设计和功能的实现可能存在不足。

（特别是算法运行结果展示和可视化分析的界面我觉得需要重新设计...）

11个PLS相关算法由学姐（两个交叉的圆）和学长（LOVE）提供并进行初步整理,

当前版本(v1.0)集成的部分算法从运行效果来看不是特别满意，

所以部分算法可能还存在很多不足，后期如果有机会将进行修改和优化。

#### 后续
**欢迎大家加入本项目的开发和维护工作!**

**欢迎为本项目提供宝贵的建议！！**

**欢迎一起学习交流！！！**

##### 联系我
1. 作者：tbyouth
2. QQ： 2638961251
3. 邮箱：<tbyouth@gmail.com> or <2638961251@qq.com>

#### 项目其他截图
![dr3Tln.png](https://s1.ax1x.com/2020/08/24/dr3Tln.png)

![dr37yq.png](https://s1.ax1x.com/2020/08/24/dr37yq.png)

![dr34YQ.png](https://s1.ax1x.com/2020/08/24/dr34YQ.png)





