#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/19 0019 15:13
#@Author  :    tb_youth
#@FileName:    SplitterExample.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

from PyQt5.Qt import (QWidget, QHBoxLayout, QFrame,
                      QSplitter,  QApplication)
from PyQt5.QtCore import Qt
import sys


class SplitterExample(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.resize(800, 800)
        self.setWindowTitle('QSplitter')
        # 布局
        hbox = QHBoxLayout(self)
        # QFrame创建和设置
        topleft = QFrame(self)
        topleft.setFrameShape(QFrame.StyledPanel)

        topright = QFrame(self)
        topright.setFrameShape(QFrame.StyledPanel)

        bottom = QFrame(self)
        bottom.setFrameShape(QFrame.StyledPanel)
        # 分隔符创建
        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(topleft)
        splitter1.addWidget(topright)

        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)

        hbox.addWidget(splitter2)
        self.setLayout(hbox)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SplitterExample()
    sys.exit(app.exec_())
