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
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QComboBox, QCompleter, QSpinBox,QDoubleSpinBox
from PyQt5.QtWidgets import QFormLayout,QRadioButton
from PyQt5.QtGui import QIcon, QRegExpValidator
from PyQt5.QtCore import pyqtSignal, QObject, QRegExp


#自定义的信号类，用于窗口通信
class MySignal(QObject):
    sender = pyqtSignal(dict)
    def send(self,parameter_dict):
        self.sender.emit(parameter_dict)



# 0,'DSA-PLS'
class Widget0(QWidget):
    def __init__(self,cnt=10000):
        super(Widget0,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,22,1000,0.5,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)


        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30','50','70','80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx,self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('主成分个数:')
        self.spinbox_2 = QSpinBox()
        #要求： 小于成分（自变量）数
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)



        self.label_3 = QLabel()
        self.label_3.setText('隐含层神经元个数:')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 100)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('迭代次数:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1,100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.label_5 = QLabel()
        self.label_5.setText('压缩参数:')
        self.spinbox_5 = QDoubleSpinBox()
        self.spinbox_5.setRange(0,1)
        self.spinbox_5.setSingleStep(0.1)
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.form_layout.addRow(self.label_5, self.spinbox_5)



        self.label_6 = QLabel()
        self.label_6.setText('步长:')
        self.label_6.setToolTip('学习率')
        self.spinbox_6 = QSpinBox()
        self.spinbox_6.setRange(1, 100)
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.form_layout.addRow(self.label_6, self.spinbox_6)

        self.label_7 = QLabel()
        self.label_7.setText('稀疏性参数:')
        self.label_7.setToolTip('通常是一个接近于0的小数')
        self.spinbox_7 = QDoubleSpinBox()
        self.spinbox_7.setRange(0, 1)
        self.spinbox_7.setDecimals(2)
        self.spinbox_7.setSingleStep(0.01)
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.form_layout.addRow(self.label_7, self.spinbox_7)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)
        self.spinbox_5.setReadOnly(state)
        self.spinbox_6.setReadOnly(state)
        self.spinbox_7.setReadOnly(state)



    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.initLineEdit(True)




    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText()) / 100,
            'n_components': self.spinbox_2.value(),
            'ny': self.spinbox_3.value(),
            'iterations': self.spinbox_4.value(),
            'beta': self.spinbox_5.value(),
            'eta': self.spinbox_6.value(),
            'sp': self.spinbox_7.value()
        }
        return parameter_dict

#1,'LAPLS'
class Widget1(QWidget):
    def __init__(self,*args):
        super(Widget1,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [80, 20,0.2000,9]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)

        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('ntest:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, 100)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('th_k:')
        # 要求 0.1000  <= th_k <= 1.0000,这里下限设置不了,后期试试正则校验
        self.spinbox_3 = QDoubleSpinBox()
        self.spinbox_3.setDecimals(4)
        self.spinbox_3.setRange(0.1000,1.0000)
        self.spinbox_3.setSingleStep(0.0001)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('lambd_k:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1, 100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)


    def initLineEdit(self,state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)

    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)

    # 设置参数可编辑
    def selfSettingParameter(self):
        self.initLineEdit(False)



    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText())/100,
            'ntest': self.spinbox_2.value(),
            'th_k': self.spinbox_3.value(),
            'lambd_k': self.spinbox_4.value()
        }
        return parameter_dict

# 2, 'RBM-PLS'
class Widget2(QWidget):
    def __init__(self,cnt=10000):
        super(Widget2,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [80,2,8,5,0.05,100,100]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)

        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('主成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('隐含层1神经元个数:')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 100)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('隐含层2神经元个数:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1, 100)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.label_5 = QLabel()
        self.label_5.setText('学习率:')
        self.spinbox_5 = QDoubleSpinBox()
        self.spinbox_5.setSingleStep(0.01)
        self.spinbox_5.setRange(0,1)
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.form_layout.addRow(self.label_5, self.spinbox_5)

        self.label_6 = QLabel()
        self.label_6.setText('batch_size:')
        self.spinbox_6 = QSpinBox()
        self.spinbox_6.setRange(1, 100)
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.form_layout.addRow(self.label_6, self.spinbox_6)

        self.label_7 = QLabel()
        self.label_7.setText('迭代次数:')
        self.spinbox_7 = QSpinBox()
        self.spinbox_7.setRange(1,10000)
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.form_layout.addRow(self.label_7, self.spinbox_7)


        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)
        self.spinbox_5.setReadOnly(state)
        self.spinbox_6.setReadOnly(state)
        self.spinbox_7.setReadOnly(state)



    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText()) / 100.0,
            'n_components': self.spinbox_2.value(),
            'n01': self.spinbox_3.value(),
            'n02': self.spinbox_4.value(),
            'alpha': self.spinbox_5.value(),
            'bs': self.spinbox_6.value(),
            'ite': self.spinbox_7.value()
        }
        return parameter_dict

# 3, 'SEA-PLS'
class Widget3(QWidget):
    def __init__(self,cnt=10000):
        super(Widget3,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [80,2,22,1000,0.50,1,0.05]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)


        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('主成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('隐含层神经元个数:')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 100)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('迭代次数:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1, 100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.label_5 = QLabel()
        self.label_5.setText('压缩参数:')
        self.spinbox_5 = QDoubleSpinBox()
        self.spinbox_5.setSingleStep(0.01)
        self.spinbox_5.setRange(0,1)
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.form_layout.addRow(self.label_5, self.spinbox_5)

        self.label_6 = QLabel()
        self.label_6.setText('步长:')
        self.label_6.setToolTip('学习率')
        self.spinbox_6 = QSpinBox()
        self.spinbox_6.setRange(1, 100)
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.form_layout.addRow(self.label_6, self.spinbox_6)

        self.label_7 = QLabel()
        self.label_7.setText('稀疏性参数:')
        self.label_7.setToolTip('通常是一个接近于0的小数')
        self.spinbox_7 = QDoubleSpinBox()
        self.spinbox_7.setRange(0, 1)
        self.spinbox_7.setDecimals(2)
        self.spinbox_7.setSingleStep(0.01)
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.form_layout.addRow(self.label_7, self.spinbox_7)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)
        self.spinbox_5.setReadOnly(state)
        self.spinbox_6.setReadOnly(state)
        self.spinbox_7.setReadOnly(state)

        # 重置参数

    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.spinbox_5.setValue(self.defuat_parameter_list[4])
        self.spinbox_6.setValue(self.defuat_parameter_list[5])
        self.spinbox_7.setValue(self.defuat_parameter_list[6])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText()) / 100,
            'n_components': self.spinbox_2.value(),
            'ny': self.spinbox_3.value(),
            'iterations': self.spinbox_4.value(),
            'beta': self.spinbox_5.value(),
            'eta': self.spinbox_6.value(),
            'sp': self.spinbox_7.value()
        }
        return parameter_dict


#4,'PLS-S-DA'
class Widget4(QWidget):
    def __init__(self,cnt=10000):
        super(Widget4,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [1,0.001,150,10]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)

        self.label_1 = QLabel('成分个数:')
        self.spinbox_1 = QSpinBox()
        self.spinbox_1.setRange(1, self.cnt)
        self.spinbox_1.setValue(self.defuat_parameter_list[0])
        self.form_layout.addRow(self.label_1, self.spinbox_1)

        self.label_2 = QLabel()
        self.label_2.setText('梯度下降的步长:')
        self.label_2.setToolTip('相当于学习率')
        self.spinbox_2 = QDoubleSpinBox()
        self.spinbox_2.setRange(0, 1)
        self.spinbox_2.setDecimals(3)
        self.spinbox_2.setSingleStep(0.001)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('最大迭代次数:')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 10000)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('折叠次数:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1, 100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        self.spinbox_1.setReadOnly(state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)

    # 重置参数
    def reSetParameter(self):
        self.spinbox_1.setValue(self.defuat_parameter_list[0])
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'h': self.spinbox_1.value(),
            'alpha': self.spinbox_2.value(),
            'maxCycles': self.spinbox_3.value(),
            'n_splits': self.spinbox_4.value()

        }
        return parameter_dict

# 5, 'DBN-PLS'
class Widget5(QWidget):
    def __init__(self,*args):
        super(Widget5,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [10,0.1,1,10]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)

        self.label_1 = QLabel('epoch:')
        self.spinbox_1 = QSpinBox()
        self.spinbox_1.setRange(1, 100)
        self.spinbox_1.setValue(self.defuat_parameter_list[0])
        self.form_layout.addRow(self.label_1, self.spinbox_1)

        self.label_2 = QLabel()
        self.label_2.setText('学习率:')
        self.label_2.setToolTip('相当于学习率')
        self.spinbox_2 = QDoubleSpinBox()
        self.spinbox_2.setRange(0, 1)
        self.spinbox_2.setDecimals(2)
        self.spinbox_2.setSingleStep(0.01)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)



        self.label_3 = QLabel()
        self.label_3.setText('k:')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 100)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('batch_size:')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1,100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
      self.spinbox_1.setReadOnly(state)
      self.spinbox_2.setReadOnly(state)
      self.spinbox_3.setReadOnly(state)
      self.spinbox_4.setReadOnly(state)


    # 重置参数
    def reSetParameter(self):
        self.spinbox_1.setValue(self.defuat_parameter_list[0])
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'pretraining_epochs': self.spinbox_1.value(),
            'pretrain_lr': self.spinbox_2.value(),
            'k': self.spinbox_3.value(),
            'batch_size': self.spinbox_4.value()

        }
        return parameter_dict


# 6, 'Mtree-PLS'
class Widget6(QWidget):
    def __init__(self,cnt=10000):
        super(Widget6, self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [80, 2, 5, 2]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)

        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('max_depth:')
        self.label_3.setToolTip('树的最大深度')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(0, 100000)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('min_samples_split:')
        self.label_4.setToolTip('最小分割样本数')
        self.spinbox_4 = QSpinBox()
        #要求大于等于2，这样才会被分割
        self.spinbox_4.setRange(2, 100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)
        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)

    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText())/100,
            'h': self.spinbox_2.value(),
            'max_depth': self.spinbox_3.value(),
            'min_samples_split': self.spinbox_4.value()

        }
        return parameter_dict

# 7, 'RFPLS'
class Widget7(QWidget):
    def __init__(self,cnt=10000):
        super(Widget7, self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.setWindowTitle('设置参数')
        self.signal = MySignal()
        self.defuat_parameter_list = [80, 2, 2, 0]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button, self.self_setting_radio_button)


        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30','50','70','80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx,self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)


        self.label_3 = QLabel()
        self.label_3.setText('n_estimators:')
        self.label_3.setToolTip('树的数量')
        self.spinbox_3 = QSpinBox()
        self.spinbox_3.setRange(1, 10000)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)


        self.label_4 = QLabel()
        self.label_4.setText('max_depth:')
        self.label_4.setToolTip('树的最大深度')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(0, 100000)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)
        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self, state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)


    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)

    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText())/100,
            'h': self.spinbox_2.value(),
            'n_estimators': self.spinbox_3.value(),
            'max_depth': self.spinbox_4.value() if  self.spinbox_4.value() else None
        }
        print(self.spinbox_4.value())
        return parameter_dict


# 8, 'PLSCF'
class Widget8(QWidget):
    def __init__(self,cnt=10000):
        super(Widget8, self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        layout = QFormLayout()
        layout.addWidget(QLabel('Tip: 此算法不需要设置参数...'))
        self.setLayout(layout)
    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
        }
        return parameter_dict


# 9, 'SBM-PLS'
class Widget9(QWidget):
    def __init__(self,cnt=10000):
        super(Widget9,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)

        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('主成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.initLineEdit(True)

        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)

    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.initLineEdit(True)



    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText()) / 100,
            'n_components': self.spinbox_2.value(),
        }
        return parameter_dict

# 10, 'GRA-PLS'
class Widget10(QWidget):
    def __init__(self,cnt=10000):
        super(Widget10,self).__init__()
        self.cnt = cnt
        self.initUI()

    def initUI(self):
        self.resize(400, 200)
        self.signal = MySignal()
        self.defuat_parameter_list = [70,2,0.5,1]

        self.form_layout = QFormLayout()

        self.defuat_radio_button = QRadioButton('默认参数')
        self.self_setting_radio_button = QRadioButton('自定义参数')
        self.form_layout.addRow(self.defuat_radio_button,self.self_setting_radio_button)

        self.label_1 = QLabel('训练集占百分比:')
        self.combox_1 = QComboBox()
        self.combox_1.setEditable(True)
        list_1 = ['30', '50', '70', '80']
        self.combox_1.setCompleter(QCompleter(list_1))
        regx = QRegExp("^[0-9]{2}$")
        validator = QRegExpValidator(regx, self.combox_1)
        self.combox_1.setValidator(validator)
        self.combox_1.addItems(list_1)
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.form_layout.addRow(self.label_1, self.combox_1)

        self.label_2 = QLabel('主成分个数:')
        self.spinbox_2 = QSpinBox()
        self.spinbox_2.setRange(1, self.cnt)
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.form_layout.addRow(self.label_2, self.spinbox_2)

        self.label_3 = QLabel()
        self.label_3.setText('压缩参数:')
        self.spinbox_3 = QDoubleSpinBox()
        self.spinbox_3.setRange(0,1)
        self.spinbox_3.setSingleStep(0.1)
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.form_layout.addRow(self.label_3, self.spinbox_3)

        self.label_4 = QLabel()
        self.label_4.setText('步长:')
        self.label_4.setToolTip('学习率')
        self.spinbox_4 = QSpinBox()
        self.spinbox_4.setRange(1, 100)
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.form_layout.addRow(self.label_4, self.spinbox_4)

        self.initLineEdit(True)


        self.setLayout(self.form_layout)

        # 关联信号
        self.defuat_radio_button.clicked.connect(self.reSetParameter)
        self.defuat_radio_button.setChecked(True)
        self.self_setting_radio_button.clicked.connect(self.selfSettingParameter)

    def initLineEdit(self,state):
        self.combox_1.setEnabled(not state)
        self.spinbox_2.setReadOnly(state)
        self.spinbox_3.setReadOnly(state)
        self.spinbox_4.setReadOnly(state)

    # 重置参数
    def reSetParameter(self):
        self.combox_1.setCurrentText(str(self.defuat_parameter_list[0]))
        self.spinbox_2.setValue(self.defuat_parameter_list[1])
        self.spinbox_3.setValue(self.defuat_parameter_list[2])
        self.spinbox_4.setValue(self.defuat_parameter_list[3])
        self.initLineEdit(True)



    def selfSettingParameter(self):
        self.initLineEdit(False)

    # 参数设置完成，发送信号并关闭设置参数对话框
    def getParameterDict(self):
        parameter_dict = {
            'q': int(self.combox_1.currentText()) / 100,
            'n_components': self.spinbox_2.value(),
            'beta': self.spinbox_3.value(),
            'eta': self.spinbox_4.value(),
        }
        return parameter_dict


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = Widget1()
    window.show()
    sys.exit(app.exec_())
