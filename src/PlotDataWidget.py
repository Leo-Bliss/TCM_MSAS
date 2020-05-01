#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/30 0030 12:49
#@Author  :    tb_youth
#@FileName:    PlotDataWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from src.DataViewGoupBox import DataViewGoupBox

class PlotDataWidget(QWidget):
    def __init__(self):
        super(PlotDataWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 800)
        self.group_box_y = DataViewGoupBox('Y(选择绘图数据)')
        self.group_box_x = DataViewGoupBox('X(选择X轴标签)')

        layout = QVBoxLayout()
        layout.addWidget(self.group_box_y)
        layout.addWidget(self.group_box_x)
        self.setLayout(layout)

    def getData(self):
        y_list = self.group_box_y.getSelectData()
        x_list = self.group_box_x.getSelectData()
        return y_list, x_list


if __name__=='__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = PlotDataWidget()
    w.show()
    sys.exit(app.exec_())