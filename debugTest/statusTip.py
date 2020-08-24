#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/6/6 0006 16:53
#@Author  :    tb_youth
#@FileName:    statusTip.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
QWidget中QAction设置setStatusTip到自定义的QStatusBar上
'''
import sys

from PyQt5.QtWidgets import QWidget, QAction, QApplication, QStatusBar
from PyQt5.QtWidgets import QVBoxLayout,QMenuBar, QMenu, QTextEdit, QPushButton
from PyQt5.QtCore import QEvent

class MyWidget(QWidget):
    def __init__(self,parent=None):
        super(MyWidget,self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.resize(800,800)
        layout = QVBoxLayout()
        self.menu_bar = QMenuBar()
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('This is a statusBar')
        self.file_menu = QMenu('File',self)
        self.open_action = QAction('open')
        self.file_menu.addAction(self.open_action)
        self.save_action = QAction('save')
        self.file_menu.addAction(self.save_action)
        self.menu_bar.addMenu(self.file_menu)
        layout.addWidget(self.menu_bar)
        layout.addWidget(QTextEdit())
        self.button = QPushButton('button')
        layout.addWidget(self.button)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
        self.button.installEventFilter(self)
        self.open_action.installEventFilter(self)
        self.open_action.hovered.connect(lambda :self.status_bar.showMessage('1111',1000))
        self.save_action.hovered.connect(lambda :self.status_bar.showMessage('222',1000))

        # self.file_menu.installEventFilter(self)

    def eventFilter(self, obj, even):
        if even.type() == QEvent.HoverEnter:
            if  obj == self.open_action:
                self.status_bar.showMessage('open')
            if  obj == self.button:
                self.status_bar.showMessage('button')
        elif even.type() == QEvent.HoverLeave:
            self.status_bar.showMessage('')
        return QWidget.eventFilter(self,obj,even)


if __name__=='__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    sys.exit(app.exec_())

