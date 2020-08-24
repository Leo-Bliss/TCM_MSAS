#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/8/24 0024 16:05
#@Author  :    tb_youth
#@FileName:    runApp.py.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


'''
运行整个项目
'''
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

from src.MainWindow import MainWindow

if __name__=='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowIcon(QIcon('./imgs/logo.ico'))
    window.show()
    sys.exit(app.exec_())
