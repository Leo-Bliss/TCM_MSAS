#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/28 0028 14:12
#@Author  :    tb_youth
#@FileName:    DataViewGoupBox.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import qApp, QGroupBox, QPushButton, QTableView
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout
from PyQt5.QtGui import QStandardItemModel

from src.MyThreads import ReaderExcelThread,WriteExcelThread

'''
这里的表格数据不支持自定义的增删改（之前写了，出现bug），
而且这里支持的话也是没有意义的，
现在及时醒悟：
生活已经那么难了，何必为难自己，
多花点时间做有意义的事情！
'''


class DataViewGoupBox(QGroupBox):
    def __init__(self,name):
        super(DataViewGoupBox, self).__init__(name)
        self.initUI()

    def initUI(self):
        self.resize(800,800)
        # table_view
        self.model = QStandardItemModel(22, 15)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('状态栏',5000)

        hlayout = QHBoxLayout()
        self.input_btn = QPushButton('导入')
        self.output_btn = QPushButton('导出')
        hlayout.addWidget(self.status_bar)
        hlayout.addWidget(self.input_btn)
        hlayout.addWidget(self.output_btn)
        hlayout.setSpacing(5)

        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addItem(hlayout)
        self.setLayout(layout)
        self.connectSignal()


    # 关联信号
    def connectSignal(self):
        self.input_btn.clicked.connect(self.triggeredOpen)
        self.output_btn.clicked.connect(self.triggeredSave)


    # 显示状态栏消息
    def showStatus(self, msg):
        self.status_bar.showMessage(msg,5000)

    # 接收线程加载的数据
    def loadData(self, model):
        print('load...')
        self.model = model
        self.table_view.setModel(self.model)
        qApp.processEvents()

    def triggeredOpen(self):
        self.status_bar.showMessage('打开文件', 5000)
        file_name, _ = QFileDialog.getOpenFileName(self, '打开文件', '../data',
                                                   'AnyFile(*.*);;xlsx(*.xlsx);;csv(*.csv);;xls(*.xls)')
        if file_name:
            try:
                # 这里线程实例化一定要实例化成员变量，否则线程容易销毁
                self.thread = ReaderExcelThread(file_name)
                self.thread.standarModel_signal.connect(self.loadData)
                self.thread.progressRate_signal.connect(self.showStatus)
                self.thread.end_signal.connect(self.thread.quit)
                self.thread.start()
            except Exception as e:
                print(e)
                pass

    def triggeredSave(self):
        self.status_bar.showMessage('保存文件', 5000)
        file_path, _ = QFileDialog.getSaveFileName(self, '保存文件', '../data',
                                                   'xlsx(*.xlsx);;xls(*.xls);;csv(*.csv)')
        if file_path == '':
            return
        # 文件中写入数据
        try:
            self.write_thread = WriteExcelThread(file_path, self.model)
            self.write_thread.start_signal.connect(self.showStatus)
            self.write_thread.end_signal.connect(self.write_thread.quit)
            self.write_thread.start()
        except Exception as e:
            print(e)

    def getCell(self,data):
        try:
            float(data)
            return float(data) if '.' in data else int(data)
        except:
            return data

    #得到选中的数据
    def getSelectData(self):
        select_indexs = self.table_view.selectedIndexes()
        data_list = [self.getCell(index.data()) for index in select_indexs if index.data()]
        return data_list




if __name__=='__main__':
    print(111)
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = DataViewGoupBox('AA')
    w.show()
    sys.exit(app.exec_())

