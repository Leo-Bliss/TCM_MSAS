#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/5 0005 21:03
#@Author  :    tb_youth
#@FileName:    ParametersWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
各个算法所对应的设置参数界面

'''
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QLineEdit
from PyQt5.QtWidgets import QFormLayout,QRadioButton
from PyQt5.QtGui import QIntValidator,QIcon,QDoubleValidator
from PyQt5.QtCore import pyqtSignal,QObject

#自定义的信号类，用于窗口通信
class MySignal(QObject):
    sender = pyqtSignal(dict)
    def send(self,parameter_dict):
        self.sender.emit(parameter_dict)

# 0,'DSA-PLS'
class Widget0(QWidget):
    def __init__(self):
        super(Widget0,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,22,1000,0.5,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('主成分个数:')
        self.label2.setToolTip('小于特征个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('隐含层神经元个数:')
        # self.label3.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('迭代次数:')
        # self.label4.setToolTip('Tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)

        self.label5 = QLabel()
        self.label5.setText('压缩参数:')
        # self.label5.setToolTip('Tip')
        self.line_edit5 = QLineEdit()
        # self.line_edit5.setPlaceholderText('使用的是多数投票法')
        self.line_edit5.setValidator(QDoubleValidator())
        self.line_edit5.setText(str(self.defuat_parameter_list[4]))
        self.form_layout.addRow(self.label5, self.line_edit5)

        self.label6 = QLabel()
        self.label6.setText('步长:')
        self.label6.setToolTip('学习率')
        self.line_edit6= QLineEdit()
        self.line_edit6.setValidator(QIntValidator())
        self.line_edit6.setText(str(self.defuat_parameter_list[5]))
        self.form_layout.addRow(self.label6, self.line_edit6)

        self.label7 = QLabel()
        self.label7.setText('稀疏性参数:')
        self.label7.setToolTip('通常是一个接近于0的小数')
        self.line_edit7 = QLineEdit()
        self.line_edit7.setValidator(QDoubleValidator())
        self.line_edit7.setText(str(self.defuat_parameter_list[6]))
        self.form_layout.addRow(self.label7, self.line_edit7)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text()) / 100.0,
            'n_components': int(self.line_edit2.text()),
            'ny': int(self.line_edit3.text()),
            'iterations': int(self.line_edit4.text()),
            'beta': float(self.line_edit5.text()),
            'eta': int(self.line_edit6.text()),
            'sp': float(self.line_edit7.text())
        }
        return parameter_dict

#1,'LAPLS'
class Widget1(QWidget):
    def __init__(self):
        super(Widget1,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [80, 20,0.2,9]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)

        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('ntest:')
        self.label2.setToolTip('tip')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('th_k:')
        self.label3.setToolTip('tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QDoubleValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('lambd_k:')
        self.label4.setToolTip('tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i, state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1,3):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    # 设置参数可编辑
    def selfSettingParameter(self):
        self.initLineEdit(False)



    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text())/100.0,
            'ntest': int(self.line_edit2.text()),
            'th_k': float(self.line_edit3.text()),
            'lambd_k': int(self.line_edit4.text())
        }
        return parameter_dict

# 2, 'RBM-PLS'
class Widget2(QWidget):
    def __init__(self):
        super(Widget2,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,8,5,0.05,100,100]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('主成分个数:')
        self.label2.setToolTip('小于特征个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('隐含层1神经元个数:')
        # self.label3.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('隐含层2神经元个数:')
        # self.label4.setToolTip('Tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)

        self.label5 = QLabel()
        self.label5.setText('学习率:')
        # self.label5.setToolTip('Tip')
        self.line_edit5 = QLineEdit()
        # self.line_edit5.setPlaceholderText('使用的是多数投票法')
        self.line_edit5.setValidator(QDoubleValidator())
        self.line_edit5.setText(str(self.defuat_parameter_list[4]))
        self.form_layout.addRow(self.label5, self.line_edit5)

        self.label6 = QLabel()
        self.label6.setText('batch_size:')
        self.label6.setToolTip('tip')
        self.line_edit6= QLineEdit()
        self.line_edit6.setValidator(QIntValidator())
        self.line_edit6.setText(str(self.defuat_parameter_list[5]))
        self.form_layout.addRow(self.label6, self.line_edit6)

        self.label7 = QLabel()
        self.label7.setText('迭代次数:')
        self.label7.setToolTip('tip')
        self.line_edit7 = QLineEdit()
        self.line_edit7.setValidator(QIntValidator())
        self.line_edit7.setText(str(self.defuat_parameter_list[6]))
        self.form_layout.addRow(self.label7, self.line_edit7)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text()) / 100.0,
            'n_components': int(self.line_edit2.text()),
            'n01': int(self.line_edit3.text()),
            'n02': int(self.line_edit4.text()),
            'alpha': float(self.line_edit5.text()),
            'bs': int(self.line_edit6.text()),
            'ite': int(self.line_edit7.text())
        }
        return parameter_dict

# 3, 'SEA-PLS'
class Widget3(QWidget):
    def __init__(self):
        super(Widget3,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,22,1000,0.5,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('主成分个数:')
        self.label2.setToolTip('小于特征个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('隐含层神经元个数:')
        # self.label3.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('迭代次数:')
        # self.label4.setToolTip('Tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)

        self.label5 = QLabel()
        self.label5.setText('压缩参数:')
        # self.label5.setToolTip('Tip')
        self.line_edit5 = QLineEdit()
        # self.line_edit5.setPlaceholderText('使用的是多数投票法')
        self.line_edit5.setValidator(QDoubleValidator())
        self.line_edit5.setText(str(self.defuat_parameter_list[4]))
        self.form_layout.addRow(self.label5, self.line_edit5)

        self.label6 = QLabel()
        self.label6.setText('步长:')
        self.label6.setToolTip('学习率')
        self.line_edit6= QLineEdit()
        self.line_edit6.setValidator(QIntValidator())
        self.line_edit6.setText(str(self.defuat_parameter_list[5]))
        self.form_layout.addRow(self.label6, self.line_edit6)

        self.label7 = QLabel()
        self.label7.setText('稀疏性参数:')
        self.label7.setToolTip('通常是一个接近于0的小数')
        self.line_edit7 = QLineEdit()
        self.line_edit7.setValidator(QDoubleValidator())
        self.line_edit7.setText(str(self.defuat_parameter_list[6]))
        self.form_layout.addRow(self.label7, self.line_edit7)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 8):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text()) / 100.0,
            'n_components': int(self.line_edit2.text()),
            'ny': int(self.line_edit3.text()),
            'iterations': int(self.line_edit4.text()),
            'beta': float(self.line_edit5.text()),
            'eta': int(self.line_edit6.text()),
            'sp': float(self.line_edit7.text())
        }
        return parameter_dict


#4,'PLS-S-DA'
class Widget4(QWidget):
    def __init__(self):
        super(Widget4,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [1,0.001,150,10]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('成分个数:')
        self.label1.setToolTip('tip')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('梯度下降的步长:')
        self.label2.setToolTip('相当于学习率')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QDoubleValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('最大迭代次数:')
        # self.label3.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('折叠次数:')
        # self.label4.setToolTip('Tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)


        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'h': int(self.line_edit1.text()),
            'alpha': float(self.line_edit2.text()),
            'maxCycles': int(self.line_edit3.text()),
            'n_splits': int(self.line_edit4.text())

        }
        return parameter_dict

# 5, 'DBN-PLS'
class Widget5(QWidget):
    def __init__(self):
        super(Widget5,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [10,0.1,1,10]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('epoch:')
        self.label1.setToolTip('tip')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('学习率:')
        self.label2.setToolTip('相当于学习率')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QDoubleValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('k:')
        # self.label3.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('batch_size:')
        # self.label4.setToolTip('Tip')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)
        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'pretraining_epochs': int(self.line_edit1.text()),
            'pretrain_lr': float(self.line_edit2.text()),
            'k': int(self.line_edit3.text()),
            'batch_size': int(self.line_edit4.text())

        }
        return parameter_dict


# 6, 'Mtree-PLS'
class Widget6(QWidget):
    def __init__(self):
        super(Widget6, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [0.8, 10, 5, 2]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('q:')
        self.label1.setToolTip('tip')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QDoubleValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('h:')
        self.label2.setToolTip('成分个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('max_depth:')
        self.label3.setToolTip('树的最大深度')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('min_samples_split:')
        self.label4.setToolTip('最小分割样本数')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)
        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i, state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': float(self.line_edit1.text()),
            'h': int(self.line_edit2.text()),
            'max_depth': int(self.line_edit3.text()),
            'min_samples_split': int(self.line_edit4.text())

        }
        return parameter_dict

# 7, 'RFPLS'
class Widget7(QWidget):
    def __init__(self):
        super(Widget7, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [0.8, 10, 10, 5]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('q:')
        self.label1.setToolTip('tip')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QDoubleValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('h:')
        self.label2.setToolTip('成分个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('n_estimators:')
        self.label3.setToolTip('树的数量')
        self.line_edit3 = QLineEdit()
        self.line_edit3.setValidator(QIntValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[2]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('max_depth:')
        self.label4.setToolTip('树的最大深度')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[3]))
        self.form_layout.addRow(self.label4, self.line_edit4)
        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i, state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': float(self.line_edit1.text()),
            'h': int(self.line_edit2.text()),
            'n_estimators': int(self.line_edit3.text()),
            'max_depth': int(self.line_edit4.text())
        }
        return parameter_dict


# 8, 'PLSCF'
class Widget8(QWidget):
    def __init__(self):
        super(Widget8, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
    #     self.signal = MySignal()
    #     self.defuat_parameter_list = [0.8, 10, 10, 5]
    #
    #     self.form_layout = QFormLayout()
    #
    #     self.defuat_radio_button = QRadioButton('默认参数')
    #     self.self_setting_radio_button = QRadioButton('自定义参数')
    #     self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)
    #     self.label1 = QLabel()
    #     self.label1.setText('q:')
    #     self.label1.setToolTip('tip')
    #     self.line_edit1 = QLineEdit()
    #     self.line_edit1.setValidator(QDoubleValidator())
    #     self.line_edit1.setText(str(self.defuat_parameter_list[0]))
    #     self.form_layout.addRow(self.label1, self.line_edit1)
    #
    #     self.label2 = QLabel()
    #     self.label2.setText('h:')
    #     self.label2.setToolTip('成分个数')
    #     self.line_edit2 = QLineEdit()
    #     self.line_edit2.setValidator(QIntValidator())
    #     self.line_edit2.setText(str(self.defuat_parameter_list[1]))
    #     self.form_layout.addRow(self.label2, self.line_edit2)
    #
    #     self.label3 = QLabel()
    #     self.label3.setText('n_estimators:')
    #     self.label3.setToolTip('树的数量')
    #     self.line_edit3 = QLineEdit()
    #     self.line_edit3.setValidator(QIntValidator())
    #     self.line_edit3.setText(str(self.defuat_parameter_list[2]))
    #     self.form_layout.addRow(self.label3, self.line_edit3)
    #
    #     self.label4 = QLabel()
    #     self.label4.setText('max_depth:')
    #     self.label4.setToolTip('树的最大深度')
    #     self.line_edit4 = QLineEdit()
    #     self.line_edit4.setValidator(QIntValidator())
    #     self.line_edit4.setText(str(self.defuat_parameter_list[3]))
    #     self.form_layout.addRow(self.label4, self.line_edit4)
    #     self.initLineEdit(True)
    #
    #     self.setLayout(self.form_layout)
    #     # 关联信号
    #     self.defuat_radio_button.clicked.connect(self.reSetParameter)
    #     self.defuat_radio_button.setChecked(True)
    #     self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)
    #
    # def initLineEdit(self, state):
    #     for i in range(1, 5):
    #         try:
    #             command = 'self.line_edit{}.setReadOnly({})'.format(i, state)
    #             eval(command)
    #         except Exception as e:
    #             print(e)
    #             pass
    #
    # # 重置参数
    # def reSetParameter(self):
    #     for i in range(1, 5):
    #         try:
    #             command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
    #             eval(command)
    #         except Exception as e:
    #             print(e)
    #     self.initLineEdit(True)
    #
    # def selfSettingParameter(self):
    #     self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
        }
        return parameter_dict


# 9, 'SBM-PLS'
class Widget9(QWidget):
    def __init__(self):
        super(Widget9,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,22,1000,0.5,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('主成分个数:')
        self.label2.setToolTip('小于特征个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 3):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 3):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text()) / 100.0,
            'n_components': int(self.line_edit2.text()),
        }
        return parameter_dict

# 9, 'SBM-PLS'
class Widget10(QWidget):
    def __init__(self):
        super(Widget10,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.setWindowIcon(QIcon('../images/参数.png'))
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,22,1000,0.5,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)
        self.label1 = QLabel()
        self.label1.setText('训练集占百分比:')
        self.label1.setToolTip('0~100')
        self.line_edit1 = QLineEdit()
        self.line_edit1.setValidator(QIntValidator())
        self.line_edit1.setText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label1, self.line_edit1)

        self.label2 = QLabel()
        self.label2.setText('主成分个数:')
        self.label2.setToolTip('小于特征个数')
        self.line_edit2 = QLineEdit()
        self.line_edit2.setValidator(QIntValidator())
        self.line_edit2.setText(str(self.defuat_parameter_list[1]))
        self.form_layout.addRow(self.label2, self.line_edit2)

        self.label3 = QLabel()
        self.label3.setText('压缩参数:')
        # self.label5.setToolTip('Tip')
        self.line_edit3 = QLineEdit()
        # self.line_edit5.setPlaceholderText('使用的是多数投票法')
        self.line_edit3.setValidator(QDoubleValidator())
        self.line_edit3.setText(str(self.defuat_parameter_list[4]))
        self.form_layout.addRow(self.label3, self.line_edit3)

        self.label4 = QLabel()
        self.label4.setText('步长:')
        self.label4.setToolTip('学习率')
        self.line_edit4 = QLineEdit()
        self.line_edit4.setValidator(QIntValidator())
        self.line_edit4.setText(str(self.defuat_parameter_list[5]))
        self.form_layout.addRow(self.label4, self.line_edit4)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setReadOnly({})'.format(i,state)
                eval(command)
            except Exception as e:
                print(e)
                pass

    # 重置参数
    def reSetParameter(self):
        for i in range(1, 5):
            try:
                command = 'self.line_edit{}.setText(str(self.defuat_parameter_list[i-1]))'.format(i)
                eval(command)
            except Exception as e:
                print(e)
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.line_edit1.text()) / 100.0,
            'n_components': int(self.line_edit2.text()),
            'beta': float(self.line_edit3.text()),
            'eta': int(self.line_edit4.text()),
        }
        return parameter_dict


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = Widget8()
    window.show()
    sys.exit(app.exec_())
