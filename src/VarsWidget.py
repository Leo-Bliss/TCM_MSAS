#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/5 0005 21:18
#@Author  :    tb_youth
#@FileName:    VarsWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QAbstractItemView
from PyQt5.QtWidgets import QListWidget, QLabel, QPushButton,QCheckBox
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
# from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QObject, pyqtSignal


class MySignal(QObject):
    sender = pyqtSignal(list)
    def send(self, var_list):
        self.sender.emit(var_list)


class VarTabWidget(QWidget):
    '''
    变量设置界面
    '''
    def __init__(self, data_list,id):
        super(VarTabWidget, self).__init__()
        self.depend_var_tip = '<font color=blue>此算法可以多因变量建模</font>' if id in [0,2,3,5] \
            else '<font color=blue>此算法仅适用于单因变量建模</font>'
        self.initUI(data_list)

    def initUI(self, data_list):
        self.resize(800, 800)
        # self.setWindowTitle('变量设置')
        self.signal = MySignal()
        self.label1 = QLabel('变量')
        self.label2 = QLabel('自变量')
        self.label3 = QLabel('因变量')
        '''
        list_widget1:全部变量列表
        list_widget2：自变量列表
        list_widget3：因变量列表
        '''
        self.list_widget1 = QListWidget()
        self.list_widget2 = QListWidget()
        self.list_widget3 = QListWidget()

        self.list_widget1.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget2.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget3.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget1.addItems([str(item) for item in data_list])

        self.button12 = QPushButton('-->')
        self.button13 = QPushButton('-->')

        vlayout1 = QVBoxLayout()
        vlayout1.addWidget(self.button12)

        hlayout1 = QHBoxLayout()
        hlayout1.addItem(vlayout1)
        vlayout = QVBoxLayout()
        top_hlayout2 = QHBoxLayout()
        self.checkBox2 = QCheckBox('全选')
        top_hlayout2.addWidget(self.label2)
        top_hlayout2.addWidget(self.checkBox2)
        top_hlayout2.addStretch()
        vlayout.addLayout(top_hlayout2)
        vlayout.addWidget(self.list_widget2)
        vlayout.setSpacing(10)
        hlayout1.addItem(vlayout)

        vlayout2 = QVBoxLayout()
        vlayout2.addWidget(self.button13)

        hlayout2 = QHBoxLayout()
        hlayout2.addItem(vlayout2)
        vlayout = QVBoxLayout()

        top_hlayout3 = QHBoxLayout()
        self.checkBox3 = QCheckBox('全选')
        top_hlayout3.addWidget(self.label3)
        top_hlayout3.addWidget(self.checkBox3)
        top_hlayout3.addWidget(QLabel(self.depend_var_tip))
        top_hlayout3.addStretch()
        vlayout.addLayout(top_hlayout3)
        vlayout.addWidget(self.list_widget3)
        vlayout.setSpacing(10)
        hlayout2.addItem(vlayout)

        gridlayout = QGridLayout()
        hlayout = QHBoxLayout()
        hlayout.setSpacing(20)

        vlayout = QVBoxLayout()
        vlayout.addItem(hlayout)
        top_hlayout1 = QHBoxLayout()
        self.checkBox1 = QCheckBox('全选')
        top_hlayout1.addWidget(self.label1)
        top_hlayout1.addWidget(self.checkBox1)
        top_hlayout1.addStretch()
        vlayout.addLayout(top_hlayout1)
        vlayout.addWidget(self.list_widget1)
        vlayout.setSpacing(10)

        gridlayout.addItem(vlayout, 1, 0, 2, 1)
        hlayout1.setSpacing(10)
        hlayout2.setSpacing(10)
        gridlayout.addItem(hlayout1, 1, 1, 1, 1)
        gridlayout.addItem(hlayout2, 2, 1, 1, 1)
        self.setLayout(gridlayout)


        self.button12.clicked.connect(self.onClickButton12)
        self.button13.clicked.connect(self.onClickButton13)

        self.list_widget1.setObjectName('list_widget1')
        self.list_widget2.setObjectName('list_widget2')
        self.list_widget3.setObjectName('list_widget3')
        tip = '按住Ctrl键可用鼠标实现多选！'
        self.list_widget1.setToolTip(tip)
        self.list_widget2.setToolTip(tip)
        self.list_widget3.setToolTip(tip)
        self.list_widget1.itemClicked.connect(self.focusList)
        self.list_widget2.itemClicked.connect(self.focusList)
        self.list_widget3.itemClicked.connect(self.focusList)
        self.list_widget1.itemSelectionChanged.connect(self.focusList)
        self.list_widget2.itemSelectionChanged.connect(self.focusList)
        self.list_widget3.itemSelectionChanged.connect(self.focusList)

        self.checkBox1.stateChanged.connect(self.checkBoxStatus)
        self.checkBox2.stateChanged.connect(self.checkBoxStatus)
        self.checkBox3.stateChanged.connect(self.checkBoxStatus)


    #鼠标在特定list_widget操作时改变按钮显示的操作方向
    def focusList(self):
        sender = self.sender().objectName()
        if sender == 'list_widget1':
            if self.button12.text() != '-->':
                self.button12.setText('-->')
            if self.button13.text() != '-->':
                self.button13.setText('-->')
        elif sender == 'list_widget2':
            if self.button12.text() != '<--':
                self.button12.setText('<--')
        else:
            if self.button13.text() != '<--':
                self.button13.setText('<--')

    def onClickButton12(self):
        if self.button12.text() == '-->':
            sender = self.list_widget1
            reciever = self.list_widget2
        else:
            sender = self.list_widget2
            reciever = self.list_widget1
        if sender is not None and reciever is not None:
            self.sendData(sender, reciever)

    def onClickButton13(self):
        if self.button13.text() == '-->':
            sender = self.list_widget1
            reciever = self.list_widget3
        else:
            sender = self.list_widget3
            reciever = self.list_widget1
        self.sendData(sender,reciever)

    def sendData(self, sender, reciever):
        try:
            item_list = sender.selectedItems()
            for item in item_list:
                reciever.addItem(item.text())
                sender.takeItem(sender.row(item))
                self.initCheckBox()
        except Exception as e:
            print(e)

    def checkBoxStatus(self):
        sender = self.sender()
        if(sender == self.checkBox1):
            if self.checkBox1.isChecked():
                self.list_widget1.selectAll()
            else:
                self.list_widget1.clearSelection()
        elif (sender == self.checkBox2):
            if self.checkBox2.isChecked():
                self.list_widget2.selectAll()
            else:
                self.list_widget2.clearSelection()
        else:
            if self.checkBox1.isChecked():
                self.list_widget3.selectAll()
            else:
                self.list_widget3.clearSelection()

    def initCheckBox(self):
        self.checkBox1.setChecked(False)
        self.checkBox2.setChecked(False)
        self.checkBox3.setChecked(False)


    def getVarDict(self):
        '''
        :return: 设置的变量
        '''
        count1 = self.list_widget2.count()
        count2 = self.list_widget3.count()
        independ_var = [self.list_widget2.item(i).text() for i in range(count1)]
        depend_var = [self.list_widget3.item(i).text() for i in range(count2)]
        if len(independ_var)==0 or len(depend_var)==0:
            return None
        var_dict = {'independ_var':independ_var, 'depend_var':depend_var}
        return var_dict


if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_list = ['1', '2', '3', '4', '5']
    window = VarTabWidget(data_list,0)
    window.show()
    sys.exit(app.exec_())