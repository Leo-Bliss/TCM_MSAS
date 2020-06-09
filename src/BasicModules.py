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
from PyQt5.QtWidgets import QHBoxLayout, QCheckBox, QLineEdit,\
    QSpinBox, QComboBox, QPushButton, QColorDialog


class CheckboxEdit(QHBoxLayout):
    '''
    自定义组件：QCheckBox + QLineEdit
    '''

    def __init__(self, name,text=''):
        super(CheckboxEdit, self).__init__()
        self.checkbox = QCheckBox(name)
        self.initUI(text)

    def initUI(self,text):
        self.line_edit = QLineEdit(text)
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
            return None

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
        num = self.getNum(self.line_edit.text())
        return num if num is not None else self.line_edit.text()

    def isChecked(self):
        return self.checkbox.isChecked()


class CheckboxSpinBox(QHBoxLayout):
    '''
        自定义组件：QCheckBox + QSpinBox
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
        自定义组件：QCheckBox + QComBox
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
        self.combox.setVisible(self.checkbox.isChecked())


    def setItems(self, items):
        self.combox.addItems(items)

    def getValue(self):
        return self.combox.currentText()

    def isChecked(self):
        return self.checkbox.isChecked()

    def setDefualtValue(self, index):
        self.combox.setCurrentIndex(index)

class LineEditButton(QHBoxLayout):
    '''
    自定义组件，QLineEdit +  QPushuButton
    主要用于颜色选择，单击按钮弹出颜色选择对话框
    '''

    def __init__(self,init_color='#59a869'):
        super(LineEditButton,self).__init__()
        self.initUI()

        self.initColor(init_color)

    def initUI(self):
        self.line_edit = QLineEdit()
        self.button = QPushButton()
        self.button.setToolTip('单击选择颜色')
        self.button.setFixedSize(12,12)
        self.addWidget(self.line_edit)
        self.addWidget(self.button)
        self.setSpacing(5)
        self.setStretch(0,4)
        self.setStretch(1,1)
        self.button.clicked.connect(self.onClickedBtn)

    def onClickedBtn(self):
        color = QColorDialog().getColor()
        color_style = 'background-color:{};border-radius:2px;'.format(color.name())
        self.button.setStyleSheet(color_style)
        self.line_edit.setText(color.name())

    def initColor(self,color):
        self.line_edit.setText(color)
        color_style = 'background-color:{};border-radius:2px;'.format(color)
        self.button.setStyleSheet(color_style)

    def getText(self):
        return self.line_edit.text()

    def hideWidget(self):
        self.line_edit.hide()
        self.button.hide()

    def showWidget(self):
        self.line_edit.show()
        self.button.show()



if __name__=='__main__':
    import sys
    from PyQt5.QtWidgets import QApplication,QWidget
    app = QApplication(sys.argv)
    window = QWidget()
    window.setLayout(LineEditButton())
    window.show()
    sys.exit(app.exec_())


