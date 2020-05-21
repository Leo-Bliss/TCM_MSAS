#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/1 0001 12:08
#@Author  :    tb_youth
#@FileName:    BriefWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


'''
运行结果摘要,
展示预测值和真实值，并且可结果导出
'''

from PyQt5.QtWidgets import QWidget, QTextEdit, QStatusBar, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QHeaderView, \
    QScrollArea, QFrame
from PyQt5.QtWidgets import QTableView
from PyQt5.QtCore import QAbstractTableModel,Qt

from src.ProgressBar import CircleProgressBar

class TableModel(QAbstractTableModel):
    def __init__(self, data, parent=None):
        super(TableModel, self).__init__(parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data[0]) if self.rowCount() else 0

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row = index.row()
            if 0 <= row < self.rowCount():
                column = index.column()
                if 0 <= column < self.columnCount():
                    return self._data[row][column]


class ShowWidget(QWidget):
    def __init__(self):
        super(ShowWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.scrollarea = QScrollArea()
        self.scrollarea.setFrameShape(QFrame.NoFrame)

        vlayout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        vlayout.addWidget(self.text_edit)
        vlayout.addWidget(QTableView())
        vlayout.setStretch(0,7)
        vlayout.setStretch(1,3)
        vlayout.setContentsMargins(0,0,0,0)
        self.scrollarea_content = QWidget()
        self.scrollarea_content.resize(500,2000)
        self.scrollarea_content.setLayout(vlayout)
        self.scrollarea.setWidget(self.scrollarea_content)
        layout = QVBoxLayout()
        layout.addWidget(self.scrollarea)
        self.scrollarea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollarea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)


class BriefWidget(QWidget):
    def __init__(self):
        super(BriefWidget,self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800,800)
        self.text_edit = QTextEdit()
        # self.text_edit.setFrameShape(QFrame.NoFrame)
        self.text_edit.setReadOnly(True)

        self.status_bar = QStatusBar()
        self.status_bar.hide()
        self.timeCnt = 0
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.text_edit)
        hlayout = QHBoxLayout()
        self.progressbar = CircleProgressBar()
        hlayout.addWidget(self.progressbar)
        self.status_label = QLabel()
        hlayout.addWidget(self.status_label)
        hlayout.addWidget(self.status_bar)
        self.termination_btn = QPushButton('终止')
        hlayout.addWidget(self.termination_btn)
        hlayout.setSpacing(10)


        vlayout.addItem(hlayout)
        vlayout.setStretch(0,9)
        vlayout.setStretch(1,1)
        self.setLayout(vlayout)


    def appendText(self,text):
        self.text_edit.append(text)

    def showTimer(self,msg):
        self.timeCnt = msg
        self.status_bar.showMessage('运行计时:{}s'.format(msg))

    def setStatus(self,status):
        self.status_label.setText(status)

    def getRunTime(self):
        return self.timeCnt



if __name__ =='__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = BriefWidget()
    w.show()
    sys.exit(app.exec_())