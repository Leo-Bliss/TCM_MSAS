#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/28 0028 14:12
#@Author  :    tb_youth
#@FileName:    DataViewGoupBox.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
QTableView + QGoupBox
'''

from PyQt5.QtWidgets import qApp, QGroupBox, QPushButton, QTableView
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout
from PyQt5.QtGui import QStandardItemModel, QStandardItem

from src.MyThreads import ReaderExcelThread,WriteExcelThread




class DataViewGoupBox(QGroupBox):
    def __init__(self,name,data_list):
        super(DataViewGoupBox, self).__init__(name)
        self.initUI()
        self.initModel(data_list)

    def initUI(self):
        self.resize(800,800)
        self.table_view = QTableView()
        # 状态栏
        self.status_bar = QStatusBar()
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

    def initModel(self,data_list):
        self.model = QStandardItemModel(22, 15)
        if data_list:
            for i, data in enumerate(data_list):
                item = QStandardItem(str(data))
                self.model.setItem(i, 0, item)
        self.table_view.setModel(self.model)

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
            self.thread = ReaderExcelThread(file_name)
            self.thread.standarModel_signal.connect(self.loadData)
            self.thread.progressRate_signal.connect(self.showStatus)
            self.thread.end_signal.connect(self.thread.quit)
            self.thread.start()


    def triggeredSave(self):
        self.status_bar.showMessage('保存文件', 5000)
        file_path, _ = QFileDialog.getSaveFileName(self, '保存文件', '../data',
                                                   'xlsx(*.xlsx);;csv(*.csv)')
        if file_path:
            self.write_thread = WriteExcelThread(file_path, self.model)
            self.write_thread.start_signal.connect(self.showStatus)
            self.write_thread.end_signal.connect(self.write_thread.quit)
            self.write_thread.start()


    #得到表格中相应的数据类型
    def getCell(self,data):
        try:
            float(data)
            return float(data) if '.' in data else int(data)
        except:
            return data


    def getSelectData(self):
        '''
        得到选中的数据：
        目前返回的绘图数据只支持绘制部分图形（折线，散点，柱状图）
        通用性不强，不适合绘制某些类型的图形（比如箱线图），待修改，
        （可参考下面代码【返回一个二维矩阵】来思考如何修改）
	    select_indexs = self.table_view.selectedIndexes()
        if select_indexs:
            rows = sorted(index.row() for index in select_indexs)
            columns = sorted(index.column() for index in select_indexs)
            row_count = rows[-1] - rows[0] + 1
            col_count = columns[-1] - columns[0] + 1
            table = [[0 for _ in range(col_count)] for _ in range(row_count)]
            for index in select_indexs:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = self.getCell(index.data())
            print(table)
            return table
        :return:
        '''
        select_indexs = self.table_view.selectedIndexes()
        data_list = [self.getCell(index.data()) for index in select_indexs if index.data()]
        return data_list




if __name__=='__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = DataViewGoupBox('AA')
    w.show()
    sys.exit(app.exec_())

