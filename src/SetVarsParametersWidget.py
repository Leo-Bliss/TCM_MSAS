#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/5 0005 20:39
#@Author  :    tb_youth
#@FileName:    SetVarsParametersWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
集成设置变量和设置参数面板
'''

import sys
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QLabel, QMessageBox
from PyQt5.QtWidgets import QTabWidget,QHBoxLayout,QVBoxLayout
from PyQt5.QtCore import pyqtSignal,QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from src import ParametersWidget as pt
from src import VarsWidget


#自定义的信号类，用于窗口通信
class MySignal(QObject):
    sender = pyqtSignal(int,dict)
    def send(self,id,parameter_dict):
        self.sender.emit(id,parameter_dict)


class PramaeterTabWidget(QTabWidget):
    def __init__(self,id,var_list):
        super(PramaeterTabWidget,self).__init__()
        self.initUI(id,var_list)

    def initUI(self,id,var_list):
        self.setWindowTitle('参数设置')
        self.resize(800,800)
        parameter_widget_dict = {
            0: pt.Widget0,
            1: pt.Widget1,
            2: pt.Widget2,
            3: pt.Widget3,
            4: pt.Widget4,
            5: pt.Widget5,
            6: pt.Widget6,
            7: pt.Widget7,
            8: pt.Widget8,
            9: pt.Widget9,
            10: pt.Widget10
        }
        self.tab1 = VarsWidget.VarTabWidget(var_list,id)
        self.tab2 = parameter_widget_dict[id](len(var_list)-1)
        self.addTab(self.tab1,'变量设置')
        self.addTab(self.tab2,'参数设置')


class SetParameterDialog(QDialog):
    def __init__(self,id,var_list,parent=None):
        super(SetParameterDialog,self).__init__(parent)
        self.initUI(id,var_list)

    def initUI(self,id,var_list):
        self.setWindowTitle('参数设置')
        self.setWindowFlags(Qt.Window|Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon('../imgs/school_logo.png'))
        self.resize(600,500)
        self.id = id
        self.tab_widegt = PramaeterTabWidget(self.id,var_list)
        self.ok_button = QPushButton('开始建模')
        self.cancel_button = QPushButton('取消')
        self.tip_label = QLabel()
        self.tip_label.setText('<font color=blue> 请设置好各选项卡中的相关参数再开始建模！</font>')
        self.tip_label.setAlignment(Qt.AlignRight)
        hlayout = QHBoxLayout()
        hlayout.addStretch(1)
        hlayout.addWidget(self.cancel_button)
        hlayout.addWidget(self.ok_button)
        hlayout.setStretch(1,1)
        hlayout.setStretch(2, 2)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.tab_widegt)
        vlayout.addWidget(self.tip_label)
        vlayout.addLayout(hlayout)
        self.setLayout(vlayout)
        self.sendSignal = MySignal()
        self.ok_button.clicked.connect(self.getParameters)
        self.cancel_button.clicked.connect(self.close)




    def getParameters(self):
        var_dict = self.tab_widegt.tab1.getVarDict()
        # 非法操作处理：变量未设置时
        if var_dict == None:
            QMessageBox.critical(self, '错误', '您设置的变量有问题，请返回检查!', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return
        self.ok_button.setEnabled(False)
        parameter_dict,parameter_name_dict = self.tab_widegt.tab2.getParameterDict()
        all_dict = {
            'var_dict':var_dict,
            'parameter_dict':parameter_dict,
            'parameter_name_dict':parameter_name_dict}
        self.sendSignal.send(self.id,all_dict)
        self.close()


if __name__=='__main__':
    app = QApplication(sys.argv)
    window = SetParameterDialog(0,['1','2','3'])
    window.show()
    sys.exit(app.exec_())
