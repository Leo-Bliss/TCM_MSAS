#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/6/5 0005 18:32
#@Author  :    tb_youth
#@FileName:    ColorDialog.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
颜色选择
'''

# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QPushButton, QColorDialog, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import sys


class ColorDialog(QWidget):
    def __init__(self):
        super().__init__()
        # 颜色值
        color = QColor(0, 0, 0)
        # 位置
        self.setGeometry(300, 300, 350, 280)
        # 标题
        self.setWindowTitle('颜色选择')
        # 按钮名称
        self.button = QPushButton('Dialog', self)
        self.button.setFocusPolicy(Qt.NoFocus)
        # 按钮位置
        self.button.move(40, 20)
        # 按钮绑定方法
        self.button.clicked.connect(self.showDialog)
        self.setFocus()
        self.widget = QWidget(self)
        self.widget.setStyleSheet('QWidget{background-color:%s} ' % color.name())
        self.widget.setGeometry(130, 22, 200, 100)

    def showDialog(self):
        col = QColorDialog.getColor()
        print(col.name())
        if col.isValid():
            self.widget.setStyleSheet('QWidget {background-color:%s}' % col.name())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qb = ColorDialog()
    qb.show()
    sys.exit(app.exec_())