#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/16 0016 19:20
#@Author  :    tb_youth
#@FileName:    MtreePLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

"""
MTree-PLS
"""
from copy import copy
import numpy as np
# from numpy import ndarray
from numpy import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score

from algorthms.SplitDataSet import SplitDataHelper


class Node:
    """
    Node class to build tree leaves.节点类来构建树叶
    Attributes:
        avg {float} -- prediction of label.预测的标签 (default: {None})
        left {Node} -- Left child node.左子节点
        right {Node} -- Right child node.右子节点
        feature {int} -- Column index.列索引
        split {int} --  Split point.分割点
        mse {float} --  Mean square error.均方误差
    """

    attr_names = ("avg", "left", "right", "feature", "split", "mse")

    def __init__(self, avg=None, left=None, right=None, feature=None, split=None, mse=None):
        self.avg = avg  # 存放预测的标签
        self.left = left  # 存放左节点
        self.right = right  # 存放右节点
        self.feature = feature  # 存放列索引（特征）
        self.split = split  # 存放分割点
        self.mse = mse  # 均方误差

    def __str__(self):  # String函数，返回该类的所有属性信息
        ret = []
        for attr_name in self.attr_names:
            attr = getattr(self, attr_name)  # 获取对象object(这里是self，也就是Node)的属性或者方法，如果存在打印出来，如果不存在，打印出默认值，默认值可选
            # Describe the attribute of Node.描述Node的属性
            if attr is None:
                continue
            if isinstance(attr, Node):  # isinstance() 函数来判断一个对象是否是一个已知的类型，这里判断attr是否是Node类型
                des = "%s: Node object." % attr_name  # 描述为Node对象
            else:
                des = "%s: %s" % (attr_name, attr)  # 描述为 属性名：属性值
            ret.append(des)

        return "\n".join(ret) + "\n"

    def copy(self, node):
        """Copy the attributes of another Node.复制另一个节点的属性
        Arguments参数:
            node {Node}
        """

        for attr_name in self.attr_names:
            attr = getattr(node, attr_name)  # 获取对象object(这里是传进来的node)的属性或者方法
            setattr(self, attr_name, attr)  # setattr() 函数对应函数 getattr()，用于设置属性值，setattr(object, name, value)


class ModelTree:
    """RegressionTree class.回归树类
    Attributes:
        root {Node} -- Root node of RegressionTree.回归树的根节点
        depth {int} -- Depth of RegressionTree.回归树深度
        _rules {list} -- Rules of all the tree nodes.所有树节点的规则
    """

    def __init__(self):
        self.root = Node()  # 回归树的根节点
        self.depth = 1  # 回归树深度
        self._rules = None  # 所有树节点的规则


    def __str__(self):  # 读取规则
        ret = []
        for i, rule in enumerate(self._rules):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            literals, avg = rule

            ret.append("Rule %d: " % i + ' | '.join(
                literals) + ' => y_hat %.4f' % avg)
        return "\n".join(ret)

    @staticmethod
    def _expr2literal(expr: list) -> str:
        """Auxiliary function of get_rules.get_rules的辅助函数
        Arguments:
            expr {list} -- 1D list like [Feature, op, split].
        Returns:
            str
        """

        feature, operation, split = expr
        operation = ">=" if operation == 1 else "<"
        return "Feature%d %s %.4f" % (feature, operation, split)

    def get_rules(self):  # 有点问题
        """Get the rules of all the tree nodes.获取所有树节点的规则。
            Expr: 1D list like [Feature, op, split].
            Rule: 2D list like [[Feature, op, split], label].
            Op: -1 means less than, 1 means equal or more than.-1表示小于，1表示大于等于。
        """

        # Breadth-First Search.广度优先搜索。
        que = [[self.root, []]]  # 队列，此时队列中只有一个元素
        self._rules = []

        while que:
            node, exprs = que.pop(0)  # exprs表达式，也即规则

            # Generate a rule when the current node is leaf node.当当前节点为叶节点时生成规则
            if not(node.left or node.right):
                # Convert expression to text.将表达式转换为文本
                literals = list(map(self._expr2literal, exprs))
                self._rules.append([literals, node.avg])

            # Expand when the current node has left child.当当前节点含有左子节点时展开
            if node.left:
                rule_left = copy(exprs)
                rule_left.append([node.feature, -1, node.split])
                que.append([node.left, rule_left])  # 往队列中添加元素

            # Expand when the current node has right child.当当前节点含有右子节点时展开
            if node.right:
                rule_right = copy(exprs)
                rule_right.append([node.feature, 1, node.split])
                que.append([node.right, rule_right])  # 往队列中添加元素

    def linearRegression(self, X, Y):  # 一元一次方程
        print(X.shape)
        X = np.ravel(X)
        Y = np.ravel(Y)
        print(X.shape, Y.shape)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        no = 0
        for i in X:
            no = no + 1
        X = mat(X)
        Y = mat(Y)
        X = X.reshape(no, 1)
        Y = Y.reshape(no, 1)  # 这两步必须弄，不然报错
        print(X.shape, Y.shape)
        print("*******************")
        lr_model = LinearRegression()
        lr_model.fit(X, Y)
        a = lr_model.coef_[0][0]  # 回归系数
        b = lr_model.intercept_[0]  # 截距
        return a, b

    def linearRegressionSSE(self, a, b, X, Y):  # 计算残差平方和
        yHat = a * X + b  # 预测值
        return sum(power(Y - yHat, 2))

    def _get_split_mse(self, col: ndarray, label: ndarray, split: float) -> Node:
        """Calculate the mse of label when col is splitted into two pieces.计算当col分裂成两块时标签的mse
        MSE as Loss fuction:
        y_hat = Sum(y_i) / n, i <- [1, n]
        Loss(y_hat, y) = Sum((y_hat - y_i) ^ 2), i <- [1, n]
        --------------------------------------------------------------------
        Arguments:
            col {ndarray} -- A feature of training data.训练数据的一个特征
            label {ndarray} -- Target values.目标值
            split {float} -- Split point of column.列分界点
        Returns:
            Node -- MSE of label and average of splitted x标签的MSE和分裂x的平均值
        """
        idx_left = (col < split)
        idx_right = (col >= split)
        label_left = label[idx_left]
        col_left = col[idx_left]
        label_right = label[idx_right]
        col_right = col[idx_right]

        # # Calculate the means of label.计算标签的均值  ！！！这是回归树的做法
        # avg_left = label_left.mean()
        # avg_right = label_right.mean()

        # 模型树的做法（以线性回归方程代替均值，也就是一元一次方程）
        coef_left, b_left = self.linearRegression(col_left, label_left)  # 保存的是回归方程的回归系数
        coef_right, b_right = self.linearRegression(col_right, label_right)

        # # Calculate the mse of label.计算标签的mse  ！！！ 回归树的做法
        # mse = (((label_left - avg_left) ** 2).sum() +
        #        ((label_right - avg_right) ** 2).sum()) / len(label)

        # 模型树做法
        mse = (self.linearRegressionSSE(coef_left, b_left, col_left, label_left) + self.linearRegressionSSE(coef_right, b_right, col_right, label_right)) / len(label)

        # Create nodes to store result.创建节点来存储结果
        avg_coef, avg_b = self.linearRegression(col, label)
        node = Node(avg=[avg_coef, avg_b], split=split, mse=mse)
        # node = Node(split=split, mse=mse)
        node.left = Node(avg=[coef_left, b_left])  # 左节点（存放左子树的索引）avg_left传入第一个参数
        node.right = Node(avg=[coef_right, b_right])  # 右节点（存放右子树的索引）

        return node

    def _choose_split(self, col: ndarray, label: ndarray) -> Node:
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get minimum mse.
        为某个特征确定划分值，迭代每个xi并将x y分成两部分，当我们得到最小mse时，最好的分割点是xi。
        Arguments:参数
            col {ndarray} -- A feature of training data.训练数据的一个特征。
            label {ndarray} -- Target values.目标值
        Returns:
            Node -- The best choice of mse, split point and average.最佳选择mse，平均和分裂点
        """
        # Feature cannot be splitted if there's only one unique element.如果只有一个唯一的元素，则不能拆分特性。
        node = Node()
        # unique = set(col)
        unique = set(np.ravel(col))
        if len(unique) == 1:
            return node

        # In case of empty split.以防空分拆
        unique.remove(min(unique))

        # Get split point which has min mse.得到最小MSE的分界点
        # map() 会根据提供的函数对指定序列做映射。map(function, iterable, ...)
        ite = map(lambda x: self._get_split_mse(col, label, x), unique)  # 例如map函数将list元素都乘以2，x=[1,2,3,4,5]print map(lambda y:y*2,x)，输出：[2, 4, 6, 8, 10]
        node = min(ite, key=lambda x: x.mse)  # 找出MSE值最小的结点，这个结点里面有某特征MSE，最佳的划分点,左节点：包括左节点avg，右节点:包括右节点avg

        return node

    def _choose_feature(self, data: ndarray, label: ndarray) -> Node:
        """Choose the feature which has minimum mse.选择mse最小的特征
        Arguments:
            data {ndarray} -- Training data. 训练数据
            label {ndarray} -- Target values. 目标值
        Returns:
            Node -- feature number, split point, average. 特征编号，分割点，平均值
        """

        # Compare the mse of each feature and choose best one.比较每个特征的mse，并选择最佳的一个。
        # map(func, seq1[, seq2,…]) map()函数是将func作用于seq中的每一个元素，并将所有的调用的结果作为一个list返回
        # _ite = map(lambda x: (self._choose_split(data[:, x], label), x), range(data.shape[1]))  # 猜想返回的是: [(node0,0),(node1,1),...]这种形式
        _ite_list = []
        for x in range(data.shape[1]):
            node_x = self._choose_split(data[:, x], label)  # 计算每个特征的最佳分割点
            if node_x.split is not None:  # 过滤掉不符合条件的元素
                _ite_list_x = [node_x, x]
                _ite_list.append(_ite_list_x)


        # ite = filter(lambda x: x[0].split is not None, _ite)  # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
        # 不知道为什么，可迭代对象ite有些问题（min(list_ite, key=lambda x: x[0].mse)取不到值），换成list_ite可以取到值
        # node = Node()
        # feature = None
        # list_ite = []
        # for i in ite:
        #     # print(i[0].__str__())
        #     # list_ite_i = (i[0], i[1])
        #     # list_ite.append(list_ite_i)
        #     if i[1] == 0:
        #         node, feature = i[0], i[1]
        #     else:
        #         if node.mse < pro_mse:
        #             node, feature = i[0], i[1]
        #     pro_mse = node.mse



        # Return None if no feature can be splitted.找到所有特征中MSE最小的特征，如果不能拆分任何特征，则返回None。default=(Node(), None)
        node, feature = min(_ite_list, key=lambda x: x[0].mse, default=(Node(), None))  # default=(Node(), None)：意思是不用拆分任何特征的情况下，node=Node(), feature=None
        node.feature = feature
        return node  # 这个node中有feature，split，MSE，left：包括左节点avg，right：包括右节点avg

    def fit(self, data: ndarray, label: ndarray, max_depth=5, min_samples_split=2):
        """Build a regression decision tree.建立一个回归决策树
        Note:
            At least there's one column in data has more than 2 unique elements,数据中至少有一列有两个以上的惟一元素
            and label cannot be all the same value.标签不能都是相同的值
        Arguments:
            data {ndarray} -- Training data.训练数据
            label {ndarray} -- Target values.目标值
        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {5})树的最大深度
            min_samples_split {int} -- The minimum number of samples required to split an internal node. (default: {2})分割内部节点所需的最小样本数
        """
        data = np.array(data)
        label = np.array(label)
        data_col_number = data.shape[1]
        # Initialize with depth, node, indexes.初始化深度，节点，索引。
        # self.root.avg = label.mean()  # ！！！这是回归树的做法
        que = [(self.depth + 1, self.root, data, label)]  # 队列，此时队列中只有一个元素

        # Breadth-First Search.广度优先搜索
        while que: #and data_col_number != 0:
            depth, node, _data, _label = que.pop(0)

            # Terminate loop if tree depth is more than max_depth.如果树深度大于max_depth，则终止循环。
            if depth > max_depth:
                depth -= 1
                break

            # Stop split when number of node samples is less than min_samples_split or Node is 100% pure.
            # 当节点样本数小于min_samples_split或节点为100%纯时停止分割
            if len(_label) < min_samples_split or all(_label == label[0]):
                continue

            # Stop split if no feature has more than 2 unique elements.如果没有超过2个唯一元素，停止分割。
            _node = self._choose_feature(_data, _label)
            if _node.split is None:
                continue

            # # 如果没有左右节点，说明是叶子结点，不用分割
            # while _node.left is None and _node.right is None:
            #     continue

            # # 如果样本的特征分裂完毕，退出，即当样本的特征数目小于树的深度时需要做的操作
            # data_col_number = data_col_number - 1

            # Copy the attributes of _node to node.将_node的属性复制到node。
            node.copy(_node)  # node的属性值和_node一致
            # print("===================================")
            # print(node.__str__())
            # print("===================================")
            # if self.depth == 1:
            #     avg_coef, avg_b = self.linearRegression(_data, _label)
            #     node.avg = [avg_coef, avg_b]

            # Put children of current node in que.将当前节点的子节点放入que中。
            idx_left = (_data[:, node.feature] < node.split)  # feature值小于分割值的样本索引
            idx_right = (_data[:, node.feature] >= node.split)  # feature值大于等于分割值的样本索引
            que.append(
                (depth + 1, node.left, _data[idx_left], _label[idx_left]))
            que.append(
                (depth + 1, node.right, _data[idx_right], _label[idx_right]))

        # Update tree depth and rules.更新树的深度和规则。
        self.depth = depth
        # self.get_rules()

    def predict_one(self, row: ndarray) -> float:
        """Auxiliary function of predict.预测的辅助功能。
        Arguments:
            row {ndarray} -- A sample of testing data.测试数据的一个样本。
        Returns:
            float -- Prediction of label.预测的标签
        """
        row = np.ravel(row)
        # row = row[0]
        node = self.root
        # print("root", node.__str__())
        father_index_node = 0  # 最终叶节点的父节点的索引
        while node.left and node.right:
            # print("feature", node.feature)
            # print("left", node.left.avg)
            # print("right", node.right.avg)
            father_index_node = node.feature
            if row[node.feature] < node.split:
                node = node.left
            else:
                node = node.right
        # node有回归系数和截距，但是没有对应的特征，深度为5的叶子结点的回归系数和截距，是通过深度为4的父节点计算的，因此，需要找到node的父节点
        # print(node.__str__())
        row_predict = row[father_index_node] * node.avg[0] + node.avg[1]  # node.feature是叶子结点对应的特征，回归系数便是这个特征和y求出来的,node.avg是一个回归系数
        # print(row_predict)
        return row_predict

    def predict(self, data: ndarray) -> ndarray:
        """Get the prediction of label.得到标签的预测。
        Arguments:
            data {ndarray} -- Testing data.测试数据
        Returns:
            mat -- Prediction of label.预测的标签
        """
        y_pre = np.apply_along_axis(self.predict_one, 1, data)  # 对数组（data）里的每一个元素进行变换，得到目标的结果。
        no = 0
        for i in y_pre:
            no = no + 1
        y_pre = mat(y_pre).reshape(no, 1)
        return y_pre
class MTree_PLS:
    def __init__(self, h, max_depth=5, min_samples_split=2):
        self.h = h
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = ModelTree()

    # 数据标准化
    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        return e0, f0

    def getRRandRMSE(self, y0, y0_predict):
        # row = shape(y0)[0]
        # mean_y = mean(y0, 0)
        # y_mean = tile(mean_y, (row, 1))
        # SSE = sum(sum(power((y0 - y0_predict), 2), 0))
        # SST = sum(sum(power((y0 - y_mean), 2), 0))
        # SSR = sum(sum(power((y0_predict - y_mean), 2), 0))
        # RR = SSR / SST
        # RMSE = sqrt(SSE / row)
        RR = r2_score(y0, y0_predict)
        RMSE = np.sqrt(mean_squared_error(y0, y0_predict))
        return RR, RMSE

    # PLS核心函数
    def PLS(self, x0, y0):
        e0 = mat(x0)
        f0 = mat(y0)
        f0_original = f0  # 原始f0，求残差时使用
        m = shape(x0)[1]
        ny = shape(y0)[1]
        w = mat(zeros((m, self.h)))
        w_star = mat(zeros((m, self.h)))
        chg = mat(eye((m)))
        my = shape(x0)[0]
        ss = mat(zeros((m, 1))).T
        t = mat(zeros((my, self.h)))  # (n, h)
        alpha = mat(zeros((m, self.h)))
        # press_i = mat(zeros((1, my)))
        # press = mat(zeros((1, m)))
        # Q_h2 = mat(zeros((1, m)))
        # beta = mat(zeros((1, m))).T
        h = self.h + 1
        for i in range(1, h):
            # 计算w,w*和t的得分向量
            matrix = e0.T * f0 * (f0.T * e0)
            val, vec = linalg.eig(matrix)  # 求特征向量和特征值
            sort_val = argsort(val)
            index_vec = sort_val[:-2:-1]
            w[:, i - 1] = vec[:, index_vec]  # 求最大特征值对应的特征向量
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
            # beta[i - 1, :] = (t[:, i - 1].T * f0) / (t[:, i - 1].T * t[:, i - 1])
            # cancha = f0 - t * beta
            if i == 1:
                t_ = t[:, i - 1]
                # continue
            else:
                t_ = t[:, 0:i]
            # print("===============================", t_.shape)  # 出现了两次一维(已改)
            self.model.fit(t_, f0, max_depth=self.max_depth, min_samples_split=self.min_samples_split)  # 用模型树计算成分t对y的回归（建模）
            temp_pre = self.model.predict(t_)
            f0 = f0_original - self.model.predict(t_)  # 求残差（预测）
            # ss[:, i - 1] = sum(sum(power(cancha, 2), 0), 1)  # 注：对不对？？？
        return w_star, t#, beta

    def train(self, x0, y0):
        x0 = mat(x0,dtype=np.float64)
        y0 = mat(y0,dtype=np.float64)
        self.m = shape(x0)[1]
        self.n = shape(y0)[1]  # 自变量和因变量个数
        row = shape(x0)[0]

        self.w_star, self.t = self.PLS(x0, y0)

        # # 求可决系数和均方根误差
        y_tr_predict = self.model.predict(self.t)  # 仅针对于训练集进行预测，因为这里的t是训练集得到的
        # print("y_tr_predict", list(y_tr_predict))
        # print("y0", list(y0))
        y_tr_RR, y0_tr_RMSE = self.getRRandRMSE(y0, y_tr_predict)

        return y_tr_predict, y_tr_RR, y0_tr_RMSE

    def predict(self, x0, y0):  # 可预测训练集和测试集
        x0 = mat(x0)
        y0 = mat(y0)
        # 先根据w_star矩阵求出t矩阵, 再由t预测y
        t = x0 * self.w_star
        y0_predict = self.model.predict(t)
        # print("y0_predict", list(y0_predict))
        # print("y0", list(y0))
        y0_RR, y0_RMSE = self.getRRandRMSE(y0, y0_predict)
        return y0_predict, y0_RR, y0_RMSE

class RunMtreePLS:
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
        self.h = parameter_dict.get('h')
        self.max_depth = parameter_dict.get('max_depth')
        self.min_samples_split = parameter_dict.get('min_samples_split')

    def run(self):
        self.initParameter()
        X = self.df[self.independent_var]
        y = self.df[self.dependent_var]
        split_helper = SplitDataHelper()
        train_x, train_y, test_x, test_y = split_helper.splitDataSet(X.values, y.values.reshape(X.shape[0], 1), q=self.q)

        # 步骤:3：建模
        """
        MTree_PLS(h, max_depth=5, min_samples_split=2)
        h = 10  # 成分个数，默认为10，其应小于等于样本的维度，从前台传进来应注意
        max_depth = 5  # 树的最大深度， 默认None，节点被扩展，直到所有叶子都是纯的（可自行设置）
        min_samples_split = 2  # 最小分割样本数，子空间样本数大于等于2才进行分割，（默认2，可自行设置，但不宜过大）
        """
        print(X.shape[1])
        assert self.h < X.shape[1], '成分个数h不能大于样本的维度'  # 传进来的成分个数大于样本的维度时，抛出异常
        mtpls_model = MTree_PLS(self.h, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        y1_predict, y_RR, y_RMSE = mtpls_model.train(train_x, train_y)  # 训练
        print(y_RMSE)

        # 预测测试集
        y_te_predict, y_te_RR, y_te_RMSE = mtpls_model.predict(test_x, test_y)
        print("测试集", y_te_RMSE)
        self.res_dict = {
            '训练集RMSE': y_RMSE,
            '测试集RMSE': y_te_RMSE
        }

    def getRes(self):
        return self.res_dict

if __name__ == '__main__':
    # 步骤1：读取数据
    # 1.1 blogData_test：1个因变量
    df_xy = pd.read_csv('../data/blogData_test1.csv')
    # print(df_xy)
    # print(df_xy.shape)
    xname_list = df_xy.columns.values.tolist()[0:df_xy.shape[1] - 1]
    #
    # X = df_xy[xname_list]
    # print(X.shape)
    # y = df_xy['y']
    var_dict = {
        'independ_var':  xname_list,
        'depend_var': ['y']
    }
    parameter_dict = {
        'q': 0.8,
        'h': 10,
        'max_depth': 5,
        'min_samples_split': 2
    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunMtreePLS(df_xy, all_dict)
    r.run()

    # # 步骤2：划分训练集测试集
    # train_x, train_y, test_x, test_y = splitDataSet(X.values, y.values.reshape(X.shape[0], 1), q=0.8)
    #
    # # 步骤:3：建模
    # """
    # MTree_PLS(h, max_depth=5, min_samples_split=2)
    # h = 10  # 成分个数，默认为10，其应小于等于样本的维度，从前台传进来应注意
    # max_depth = 5  # 树的最大深度， 默认None，节点被扩展，直到所有叶子都是纯的（可自行设置）
    # min_samples_split = 2  # 最小分割样本数，子空间样本数大于等于2才进行分割，（默认2，可自行设置，但不宜过大）
    # """
    # h = 10
    # print(X.shape[1])
    # assert h < X.shape[1], '成分个数h不能大于样本的维度'  # 传进来的成分个数大于样本的维度时，抛出异常
    # mtpls_model = MTree_PLS(h, max_depth=5, min_samples_split=2)
    # y1_predict, y_RR, y_RMSE = mtpls_model.train(train_x, train_y)  # 训练
    # print(y_RMSE)
    #
    # # 预测测试集
    # y_te_predict, y_te_RR, y_te_RMSE = mtpls_model.predict(test_x, test_y)
    # print("测试集", y_te_RMSE)

