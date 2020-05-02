#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/1 0001 12:08
#@Author  :    tb_youth
#@FileName:    BriefWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


'''
运行结果摘要
'''

from PyQt5.QtWidgets import QWidget, QTextEdit, QStatusBar, QVBoxLayout, QHBoxLayout, QLabel
from src.ProgressBar import CircleProgressBar

class BriefWidget(QWidget):
    def __init__(self):
        super(BriefWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800,800)
        self.text_edit = QTextEdit()
        self.status_bar = QStatusBar()
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.text_edit)
        hlayout = QHBoxLayout()
        self.progressbar = CircleProgressBar()
        hlayout.addWidget(self.progressbar)
        self.status_label = QLabel()
        hlayout.addWidget(self.status_label)
        hlayout.addWidget(self.status_bar)
        hlayout.setSpacing(10)

        vlayout.addItem(hlayout)
        vlayout.setStretch(0,10)
        vlayout.setStretch(1,1)
        self.setLayout(vlayout)

    def appendText(self,text):
        self.text_edit.append(text)

    def showTimer(self,msg):
        self.status_bar.showMessage(msg)

    def setStatus(self,status):
        self.status_label.setText(status)



if __name__ =='__main__':
    from PyQt5.QtWidgets import QApplication
    import e
    app = QApplication(sys.argv)
    w = BrifWidget()
    w.show()
    sys.exit(app.exec_())