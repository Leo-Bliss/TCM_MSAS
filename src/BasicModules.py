#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/30 0030 0:11
#@Author  :    tb_youth
#@FileName:    BasicModules.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
自定义的基础组件
'''
from PyQt5.QtWidgets import QHBoxLayout, QCheckBox, QLineEdit, QSpinBox, QComboBox


class CheckboxEdit(QHBoxLayout):
    '''
    自定义组件：QcheckBox + QLineEdit
    '''

    def __init__(self, name):
        super(CheckboxEdit, self).__init__()
        self.checkbox = QCheckBox(name)
        self.initUI()

    def initUI(self):
        self.line_edit = QLineEdit('')
        self.line_edit.setVisible(self.checkbox.isChecked())
        self.addWidget(self.checkbox)
        self.addWidget(self.line_edit)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.checkbox.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)

    def changeCheckboxStatus(self):
        if not self.checkbox.isChecked():
            self.setValue('')
        self.line_edit.setVisible(self.checkbox.isChecked())

    def setHolderText(self,value):
        self.line_edit.setPlaceholderText(str(value))

    def getNum(self,num):
        try:
            float(num)
            return float(num) if '.' in num else int(num)
        except:
            return 0

    #输入的是一个范围
    def getRange(self):
        l,r = 0,0
        if ','in self.line_edit.text():
            l,r = self.line_edit.text().split(',',maxsplit=1)
        elif '，'in self.line_edit.text():
            l,r = self.line_edit.text().split('，',maxsplit=1)
        return self.getNum(l),self.getNum(r)

    def setValue(self, value):
        self.line_edit.setText(str(value))

    def getValue(self):
        try:
            return self.getNum(self.line_edit.text())
        except:
            return self.line_edit.text()

    def isChecked(self):
        return self.checkbox.isChecked()


class CheckboxSpinBox(QHBoxLayout):
    '''
        自定义组件：QcheckBox + QSpinBox
    '''

    def __init__(self, name):
        super(CheckboxSpinBox, self).__init__()
        self.checkbox = QCheckBox(name)
        self.initUI()

    def initUI(self):
        self.spinbox = QSpinBox()
        self.spinbox.setVisible(self.checkbox.isChecked())
        self.spinbox.setRange(-2147483647, 2147483647)
        self.addWidget(self.checkbox)
        self.addWidget(self.spinbox)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.checkbox.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)

    def changeCheckboxStatus(self):
        self.spinbox.setVisible(self.checkbox.isChecked())


    def setValue(self, value):
        self.spinbox.setValue(value)

    def getValue(self):
        return self.spinbox.value() if self.checkbox.isChecked() else None

    def setRange(self, l, r):
        self.spinbox.setRange(l, r)

    def isChecked(self):
        return self.checkbox.isChecked()


class CheckboxComBox(QHBoxLayout):
    '''
        自定义组件：QcheckBox + QComBox
    '''

    def __init__(self, name):
        super(CheckboxComBox, self).__init__()
        self.checkbox = QCheckBox(name)
        self.initUI()

    def initUI(self):
        self.combox = QComboBox()
        self.combox.setVisible(self.checkbox.isChecked())
        self.combox.setEditable(True)
        # self.combox.setEnabled(False)
        self.addWidget(self.checkbox)
        self.addWidget(self.combox)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.checkbox.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)
    def changeCheckboxStatus(self):
        # self.combox.setEnabled(self.checkbox.isChecked())
        self.combox.setVisible(self.checkbox.isChecked())


    def setItems(self, items):
        self.combox.addItems(items)

    def getValue(self):
        return self.combox.currentText()

    def isChecked(self):
        return self.checkbox.isChecked()

    def setDefualtValue(self, index):
        self.combox.setCurrentIndex(index)