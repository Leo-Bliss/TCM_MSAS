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

from PyQt5.QtWidgets import QApplication, QWidget, QToolBar, QAction, QPushButton, QGridLayout
from PyQt5.QtWidgets import QSizePolicy,QLineEdit,QHBoxLayout
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
        self.line_edit_find.setClearButtonEnabled(True)
        self.line_edit_find.setPlaceholderText('find')
        self.line_edit_replace = QLineEdit()
        self.line_edit_replace.setClearButtonEnabled(True)
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
        self.repalce_button.setToolTip('替换当前匹配项')
        self.repalceAll_button = QPushButton('ReplaceAll')
        self.repalceAll_button.setToolTip('替换所有匹配项')
        self.search_button = QPushButton()

        gridlayout = QGridLayout()
        gridlayout.addWidget(self.line_edit_find,0,0,1,8)
        gridlayout.addWidget(self.tool_bar,0,9)
        gridlayout.addWidget(self.line_edit_replace,1,0,1,8)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.repalce_button)
        hlayout.addWidget(self.repalceAll_button)
        gridlayout.addItem(hlayout,1,9)
        hlayout.setSpacing(5)
        gridlayout.setVerticalSpacing(1)
        gridlayout.setHorizontalSpacing(5)
        self.setLayout(gridlayout)

        self.close_aciton.triggered.connect(self.triggeredClose)

        icon = QIcon()
        icon.addPixmap(QPixmap('../imgs/查找.png'), QIcon.Normal, QIcon.Off)
        self.search_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/向下.png'), QIcon.Normal, QIcon.Off)
        self.down_aciton.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/向上.png'), QIcon.Normal, QIcon.Off)
        self.up_aciton.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/关闭.png'), QIcon.Normal, QIcon.Off)
        self.close_aciton.setIcon(icon)

    def triggeredClose(self):
        self.hide()





if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    window = FindWidget()
    window.show()
    sys.exit(app.exec_())