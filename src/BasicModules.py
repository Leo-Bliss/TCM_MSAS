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
        self.line_edit = QLineEdit()
        self.defualt_value = ''
        self.line_edit.setEnabled(False)
        self.addWidget(self.checkbox)
        self.addWidget(self.line_edit)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.line_edit.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)

    def changeCheckboxStatus(self):
        self.line_edit.setEnabled(self.checkbox.isChecked())

    def setValue(self, value):
        self.line_edit.setText(str(value))

    def getValue(self):
        return self.line_edit.text() if self.checkbox.isChecked() else self.defualt_value

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
        self.spinbox.setRange(-2147483647, 2147483647)
        self.spinbox.setEnabled(False)
        self.addWidget(self.checkbox)
        self.addWidget(self.spinbox)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.spinbox.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)

    def changeCheckboxStatus(self):
        self.spinbox.setEnabled(self.checkbox.isChecked())

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
        self.combox.setEditable(True)
        self.combox.setEnabled(False)
        self.addWidget(self.checkbox)
        self.addWidget(self.combox)
        self.setStretch(0, 1)
        self.setStretch(1, 5)
        self.combox.setToolTip('勾选前方选项即可编辑')
        self.checkbox.stateChanged.connect(self.changeCheckboxStatus)

    def changeCheckboxStatus(self):
        self.combox.setEnabled(self.checkbox.isChecked())

    def setItems(self, items):
        self.combox.addItems(items)

    def getValue(self):
        return self.combox.currentText()

    def isChecked(self):
        return self.checkbox.isChecked()

    def setDefualtValue(self, index):
        self.combox.setCurrentIndex(index)