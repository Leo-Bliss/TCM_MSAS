#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/3 0003 21:36
#@Author  :    tb_youth
#@FileName:    FindWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
查找,替换窗口
'''

from PyQt5.QtWidgets import QApplication,QWidget,QToolBar,QAction,QPushButton
from PyQt5.QtWidgets import QVBoxLayout,QSizePolicy,QLineEdit,QHBoxLayout
from PyQt5.QtGui import QPixmap,QIcon,QKeySequence
from PyQt5.QtCore import Qt


class  FindWidget(QWidget):
    def __init__(self):
        super(FindWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(600,100)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)

        self.line_edit_find = QLineEdit()
        self.line_edit_find.setPlaceholderText('find')
        self.line_edit_replace = QLineEdit()
        self.line_edit_replace.setPlaceholderText('replace')

        self.tool_bar = QToolBar()
        self.search_action = QAction('查找')
        self.search_action.setShortcut(QKeySequence(QKeySequence(Qt.Key_Return)))
        self.up_aciton = QAction('向上')
        self.down_aciton = QAction('向下')
        self.close_aciton= QAction('关闭')
        self.tool_bar.addAction(self.search_action)
        self.tool_bar.addAction(self.down_aciton)
        self.tool_bar.addAction(self.up_aciton)
        self.tool_bar.addAction(self.close_aciton)

        self.repalce_button = QPushButton('Replace')
        self.repalceAll_button = QPushButton('ReplaceAll')

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.line_edit_find)
        hlayout1.addWidget(self.tool_bar)
        hlayout1.setStretch(0,5)
        hlayout1.setStretch(1,2)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.line_edit_replace)
        hlayout2.addWidget(self.repalce_button)
        hlayout2.addWidget(self.repalceAll_button)
        hlayout2.setStretch(0, 5)
        hlayout2.setStretch(1, 1)
        hlayout2.setStretch(2, 1)
        hlayout2.setSpacing(5)

        layout = QVBoxLayout()
        layout.addItem(hlayout1)
        layout.addItem(hlayout2)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.close_aciton.triggered.connect(self.triggeredClose)
        icon = QIcon()
        icon.addPixmap(QPixmap('../images/查找.png'), QIcon.Normal, QIcon.Off)
        self.search_action.setIcon(icon)
        icon.addPixmap(QPixmap('../images/向下.png'), QIcon.Normal, QIcon.Off)
        self.down_aciton.setIcon(icon)
        icon.addPixmap(QPixmap('../images/向上.png'), QIcon.Normal, QIcon.Off)
        self.up_aciton.setIcon(icon)
        icon.addPixmap(QPixmap('../images/关闭.png'), QIcon.Normal, QIcon.Off)
        self.close_aciton.setIcon(icon)

    def triggeredClose(self):
        self.hide()





if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    window = FindWidget()
    window.show()
    sys.exit(app.exec_())