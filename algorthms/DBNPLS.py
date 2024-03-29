#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/22 0022 18:40
#@Author  :    tb_youth
#@FileName:    DBNPLS.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth



# __author__=zqx


"""
将DBN提取的主成分替代pls提取的主成分(主成分分析和典型相关分析)，放到pls回归模型中
"""
# from numpy import *
from sklearn import preprocessing
import theano
import theano.tensor as T
import pandas as pd
import numpy as np

from theano.tensor.shared_randomstreams import RandomStreams




class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        self.params = [self.W, self.b]

class RBM(object):
    def __init__(self,input=None,n_visible=None,n_hidden=None,
                 W=None,hbias=None,vbias=None,np_rng=None,
                 theano_rng=None):

        self.n_visible=n_visible
        self.n_hidden=n_hidden

        #生成随机数
        if np_rng is None:
            np_rng=np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng=RandomStreams(np_rng.randint(2**30))

        if W is None:
            initial_W=np.asarray(np_rng.uniform(
                    low=-4*np.sqrt(6./(n_hidden+n_visible)),
                    high=4*np.sqrt(6./(n_hidden+n_visible)),
                    size=(n_visible,n_hidden)),
                    dtype=theano.config.floatX)
            W=theano.shared(value=initial_W,name='W',borrow=True)

        if hbias is None:
            hbias=theano.shared(value=np.zeros(n_hidden,
                                                  dtype=theano.config.floatX),
                                name='hbias',borrow=True)
        if vbias is None:
            vbias=theano.shared(value=np.zeros(n_visible,
                                                  dtype=theano.config.floatX),
                                name='vbias',borrow=True)
        self.input=input
        if not input:
            self.input=T.matrix('input')

        self.W=W
        self.hbias=hbias
        self.vbias=vbias
        self.theano_rng=theano_rng

        self.params=[self.W,self.hbias,self.vbias]

    #计算自由能
    def free_energy(self,v_sample):
        wx_b=T.dot(v_sample,self.W)+self.hbias
        vbias_term=T.dot(v_sample,self.vbias)
        hbias_term=T.sum(T.log(1+T.exp(wx_b)),axis=1)
        return -vbias_term-hbias_term

    #定义向上传播
    def propup(self,vis):
        pre_sigmoid_activation=T.dot(vis,self.W)+self.hbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]

    #给定v单元计算h单元的函数
    def sample_h_given_v(self,v0_sample):
        pre_sigmoid_h1,h1_mean=self.propup(v0_sample)
        h1_sample=self.theano_rng.binomial(size=h1_mean.shape,
                                           n=1,p=h1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_h1,h1_mean,h1_sample]

    #定义向下传播
    def propdown(self,hid):
        pre_sigmoid_activation=T.dot(hid,self.W.T)+self.vbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]


    #给定h单元计算v单元的函数
    def sample_v_given_h(self,h0_sample):

        pre_sigmoid_v1,v1_mean=self.propdown(h0_sample)
        v1_sample=self.theano_rng.binomial(size=v1_mean.shape,n=1,p=v1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_v1,v1_mean,v1_sample]

    #从隐藏状态出发，执行一步Gibbs采样过程
    def gibbs_hvh(self,h0_sample):

        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1,v1_mean,v1_sample,
                pre_sigmoid_h1,h1_mean,h1_sample]

    #从可见状态出发，执行一步Gibbs采样过程
    def gibbs_vhv(self,v0_sample):
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1,h1_mean,h1_sample,
                pre_sigmoid_v1,v1_mean,v1_sample]

    def get_cost_updates(self,lr=0.1,persistent=None,k=1):


        #计算正项
        pre_sigmoid_ph,ph_mean,ph_sample=self.sample_h_given_v(self.input)
        #决定初始化固定链的方法:对于CD，采用全新生成隐含样本；对于PCD，从链的旧状态获得
        if persistent is None:
            chain_start=ph_sample
        else:
            chain_start=persistent

        [pre_sigmoid_nvs,nv_means,nv_samples,
         pre_sigmoid_nhs,nh_means,nh_samples],updates=\
            theano.scan(self.gibbs_hvh,
                    #下面字典中前5项为None,表示chain_start与初始状态中第六个输出量有关
                    outputs_info=[None,None,None,None,None,chain_start],
                    n_steps=k)
        #计算RBM参数的梯度,只需要从链末端采样
        chain_end=nv_samples[-1]

        cost=T.mean(self.free_energy(self.input))-T.mean(self.free_energy(chain_end))
        #因为chai_end是符号变量，而我们只根据链最末端的数据求梯度，所有指定chain_end为常数
        gparams=T.grad(cost,self.params,consider_constant=[chain_end])

        #构造更新字典
        for gparam,param in zip(gparams,self.params):
            #确保学习率lr的数据类型正确
            updates[param]=param-gparam*T.cast(lr,dtype=theano.config.floatX)

        #RBM是深度网络的一个模块时,更新perisistent
        if persistent:
            #只有persistent为共享变量时才运行
            updates[persistent]=nh_samples[-1]
            #伪似然函数是PCD的一个较好的代价函数
            monitoring_cost=self.get_pseudo_likehood_cost(updates)
        #RBM是标准网络
        else:
            #重构交叉熵是CD的一个较好的代价函数
            monitoring_cost=self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])

        h = T.nnet.sigmoid(T.dot(self.input, self.W) + self.hbias)
        reconstructed_v = T.nnet.sigmoid(T.dot(h, self.W.T) + self.vbias)
        h_last = T.nnet.sigmoid(T.dot(reconstructed_v, self.W) + self.hbias)

        return monitoring_cost,updates,h_last

    def get_reconstruction_cost(self,updates,pre_sigmoid_nv):

        cross_entropy=T.mean(T.sum(self.input*T.log(T.nnet.sigmoid(pre_sigmoid_nv))+
                                   (1-self.input)*T.log(1-T.nnet.sigmoid(pre_sigmoid_nv)),axis=1))
        return cross_entropy


class DBN(object):
    def __init__(self, np_rng, theano_rng=None, n_ins=5, hidden_layers_sizes=[4,4,4]):

        self.sigmoid_layers=[]
        self.rbm_layers=[]
        self.params=[]
        self.n_layers=len(hidden_layers_sizes)
        assert self.n_layers>0

        if not theano_rng:
            theano_rng=RandomStreams(np_rng.randint(123))

        self.x=T.matrix('x')

        for i in range(self.n_layers):
            if i==0:
                input_size=n_ins
            else:
                input_size=hidden_layers_sizes[i-1]

            if i==0:
                layer_input=self.x
            else:
                layer_input=self.sigmoid_layers[i-1].output
            sigmoid_layer=HiddenLayer(rng=np_rng,input=layer_input,n_in=input_size,
                                      n_out=hidden_layers_sizes[i],activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            rbm_layer=RBM(input=layer_input,n_visible=input_size,n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,hbias=sigmoid_layer.b,np_rng=np_rng,theano_rng=theano_rng)

            self.rbm_layers.append(rbm_layer)

    def pretraining_functions(self, x, batch_size, k):

        index=T.lscalar('index')
        learning_rate=T.scalar('lr')

        batch_begin=index*batch_size
        batch_end=batch_begin+batch_size

        pretrain_fns=[]
        last_all = []
        for rbm in self.rbm_layers:
            cost,updates,h_last=rbm.get_cost_updates(learning_rate,persistent=None,k=k)

            fn=theano.function(inputs=[index,theano.In(learning_rate,value=0.1)],
                               outputs=cost,updates=updates,
                               givens={self.x:x[batch_begin:batch_end]})
            last = theano.function(inputs=[index, theano.In(learning_rate, value=0.1)],
                                 outputs=h_last, updates=updates,
                                 givens={self.x: x[batch_begin:batch_end]})
            pretrain_fns.append(fn)
            last_all.append(last)
        return pretrain_fns,last_all


class DBN_PLS:
    def __init__(self, pretraining_epochs=10, pretrain_lr=0.1, k=1, batch_size=10):
        self.pretraining_epochs = pretraining_epochs
        self.pretrain_lr = pretrain_lr
        self.k = k
        self.batch_size = batch_size
        np_rng = np.random.RandomState(123)
        print('...building the model')
        self.dbn = DBN(np_rng, n_ins=16, hidden_layers_sizes=[10, 10, 8, 5, 5])

    # 数据标准化
    def stardantDataSet(self, x0, y0):
        e0 = preprocessing.scale(x0)
        f0 = preprocessing.scale(y0)
        return e0, f0

    def data_Mean_Std(self, x0, y0):
        mean_x = np.mean(x0, 0)
        mean_y = np.mean(y0, 0)
        std_x = np.std(x0, axis=0, ddof=1)
        std_y = np.std(y0, axis=0, ddof=1)
        return mean_x, mean_y, std_x, std_y
    # 计算反标准化之后的系数
    def Calxishu(self, e0, f0, row, mean_x, mean_y, std_x, std_y):
        x = np.hstack((e0[:, :], np.mat(np.ones((row, 1)))))
        # 计算回归系数
        xishu = np.linalg.lstsq(x, f0,rcond=None)[0]
        xishu = list(xishu)
        del xishu[-1]  # 删除常数项
        xishu = np.mat(xishu)
        m = np.shape(mean_x)[1]
        n = np.shape(mean_y)[1]
        xish = np.mat(np.zeros((m, n)))
        ch0 = np.mat(np.zeros((1, n)))
        for i in range(n):
            ch0[:, i] = mean_y[:, i] - std_y[:, i] * mean_x / std_x * xishu[:, i]
            xish[:, i] = std_y[0, i] * xishu[:, i] / std_x.T
        return ch0, xish

    def getRRandRMSE(self, y0, y0_predict):
        row = np.shape(y0)[0]
        mean_y = np.mean(y0, 0)
        y_mean = np.tile(mean_y, (row, 1))
        SSE = sum(sum(np.power((y0 - y0_predict), 2), 0))
        SST = sum(sum(np.power((y0 - y_mean), 2), 0))
        SSR = sum(sum(np.power((y0_predict - y_mean), 2), 0))
        RR = SSR / SST
        RMSE = np.sqrt(SSE / row)
        return RR, RMSE

    def DBN_train(self, x0):
        x0 = np.asarray(x0, dtype=theano.config.floatX)
        x0 = theano.shared(np.asarray(x0, dtype=theano.config.floatX), borrow=True)
        n_train_batches = round(x0.get_value(borrow=True).shape[0] / self.batch_size)  # 四舍五入

        print("...getting the pretraining functions")
        pretraining_fns, last_all = self.dbn.pretraining_functions(x=x0, batch_size=self.batch_size, k=self.k)
        print("...pretraining the model")
        # start_time = time.clock()
        for i in range(self.dbn.n_layers):
            for epoch in range(self.pretraining_epochs):
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index, lr=self.pretrain_lr))
                print("pretraining layer %i, epoch %i,cost " % (i, epoch))
                print(np.mean(c))
        # end_time = time.clock()
        # print(sys.stderr, ("The pretraining code for file " + os.path.split(__file__)[1] + " ran for %0.2fm") % (
        # (end_time - start_time) / 60.))
        tran_x = None
        for i in range(self.dbn.n_layers):
            a = []
            for batch_index in range(n_train_batches):
                a.append(last_all[i](index=batch_index, lr=self.pretrain_lr))
            tran_x = np.array(a)
        list = []
        for line in tran_x:
            for newline in line:
                list.append(newline)
        X = np.mat(list)  # 经特征提取之后，x的维度发生了变化
        print("经特征提取之后的X", X)  # 经特征提取之后的X
        return X

    def train(self, x0, y0):
        X = self.DBN_train(x0)  # x0经DBN转换之后，维度发生了变化

        e0, f0 = self.stardantDataSet(X, y0)  # 标准化
        mean_x, mean_y, std_x, std_y = self.data_Mean_Std(X, y0)
        row = np.shape(X)[0]
        self.ch0, self._coef = self.Calxishu(e0, f0, row, mean_x, mean_y, std_x, std_y)  # 反标准化
        # 求可决系数和均方根误差
        y_predict = X * self._coef + np.tile(self.ch0[0, :], (row, 1))
        y_tr_RR, y_tr_RMSE = self.getRRandRMSE(y0, y_predict)
        return y_predict, y_tr_RR, y_tr_RMSE


class RunDBNPLS:
    def __init__(self, df, all_dict):
        self.df = df
        self.all_dict = all_dict
        self.res_dict = {}

    def initParameter(self):
        var_dict = self.all_dict.get('var_dict')
        parameter_dict = self.all_dict.get('parameter_dict')
        self.independent_var = var_dict.get('independ_var')
        self.dependent_var = var_dict.get('depend_var')
        self.pretraining_epochs = parameter_dict.get('pretraining_epochs')
        self.pretrain_lr = parameter_dict.get('pretrain_lr')
        self.k = parameter_dict.get('k')
        self.batch_size = parameter_dict.get('batch_size')



    def run(self):
        self.initParameter()
        x0 = np.mat(self.df[self.independent_var],dtype=float)
        y0 = np.mat(self.df[self.dependent_var],dtype=float)
        dbn_pls_model = DBN_PLS(pretraining_epochs=self.pretraining_epochs, pretrain_lr=self.pretrain_lr, k=self.k,
                                batch_size=self.batch_size)
        y_predict, y_tr_RR, y_tr_RMSE = dbn_pls_model.train(x0, y0)

        # 仅提取了x0，y没有提取，因此y和从前一样
        for real_value,predict_value in zip(y0,y_predict):
            print(real_value,predict_value)
        # 不太确点，待修正

        if len(self.dependent_var) == 1:
            predict_test = pd.DataFrame()
            # 单因变量建模，一个DataFrame显示就足够
            dependent_str = str(self.dependent_var[0])
            predict_test['{}_预测值'.format(dependent_str)] = np.ravel(y_predict)
            predict_test['{}_真实值'.format(dependent_str)] = np.ravel(y0)
            show_data_dict = {
                '预测值和真实值': predict_test
            }
        else:
            #多因变量建模，使用两个DataFrame显示
            true_data = pd.DataFrame(y0)
            true_data.columns = self.dependent_var
            predict_data = pd.DataFrame(y_predict)
            predict_data.columns = self.dependent_var


            show_data_dict = {
                '预测值': predict_data,
                '真实值':true_data
            }

        self.res_dict = {
            '可决系数':y_tr_RR,
            '均方根误差':y_tr_RMSE,
            # '回归系数':dbn_pls_model.ch0,
            'show_data_dict': show_data_dict
        }
        print(u"可决系数:", y_tr_RR)
        print(u"均方根误差:", y_tr_RMSE)
        print(u"回归系数：")
        print(dbn_pls_model.ch0)
        # 经DBN特征提取之后，x0的维度变成5维，所以回归系数的维度5维
        print(dbn_pls_model._coef)

    def getRes(self):
        return self.res_dict



if __name__ == '__main__':
    df = pd.read_excel('../data/DBNPLS_test.xlsx')
    print(df.shape)
    headers = df.columns.values.tolist()
    var_dict = {
        'independ_var': headers[0:16],
        'depend_var': headers[16:18]
    }
    parameter_dict = {
        'pretraining_epochs': 10,
        'pretrain_lr': 0.1,
        'k': 1,
        'batch_size': 10

    }
    all_dict = {
        'var_dict': var_dict,
        'parameter_dict': parameter_dict
    }
    r = RunDBNPLS(df, all_dict)
    r.run()

    # 建模
    """
    DBN_PLS(pretraining_epochs=10, pretrain_lr=0.1, k=1, batch_size=10)
    pretraining_epochs:epoch,默认为10，可自行设置
    pretrain_lr：学习率，默认为0.1，可自行设置
    batch_size：默认为10，可自行设置
    """
