#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/19 0019 0:02
#@Author  :    tb_youth
#@FileName:    ModeResWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QStatusBar, QFileDialog, QTextEdit
from PyQt5.QtWidgets import QTableView
from PyQt5.QtCore import QAbstractTableModel
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from src.MyThreads import WriteExcelThread
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



class TableViewWidget(QWidget):
    def __init__(self,*args):
        super(TableViewWidget,self).__init__()
        self.initUI(*args)

    def initUI(self,*args):
        self.data_name = QLabel()
        self.table_view = QTableView()
        self.output_btn = QPushButton('导出')
        self.name, self.data_list = args
        self.data_name.setText(self.name)
        self.model = TableModel(self.data_list, self.table_view)
        self.table_view.setModel(self.model)
        # self.table_view.verticalHeader().setVisible(False)
        #self.table_view.horizontalHeader().setVisible(False)
        # 文本自适应
        self.table_view.resizeColumnsToContents()
        # 水平填满
        # self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 垂直填满
        # self.table_view.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.status_bar = QStatusBar()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.status_bar)
        hlayout.addWidget(self.output_btn)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.data_name)
        vlayout.addWidget(self.table_view)
        vlayout.addItem(hlayout)

        self.setLayout(vlayout)

        self.output_btn.clicked.connect(self.onClickedOutPutBtn)


    # 显示状态栏消息
    def showStatus(self, msg):
        self.status_bar.showMessage(msg)

    def onClickedOutPutBtn(self):
        self.status_bar.showMessage('保存文件', 5000)
        file_path, _ = QFileDialog.getSaveFileName(self, '保存文件', '../data',
                                                   'xlsx(*.xlsx);;csv(*.csv)')
        if file_path == '':
            return

        self.write_thread = WriteExcelThread(file_path, self.data_list)
        self.write_thread.start_signal.connect(self.showStatus)
        self.write_thread.end_signal.connect(self.write_thread.quit)
        self.write_thread.start()



class RunStatusWidget(QWidget):
    def __init__(self):
        super(RunStatusWidget,self).__init__()
        self.initUI()

    def initUI(self):
        vlayout = QVBoxLayout()
        vlayout.addWidget(CircleProgressBar())
        self.label = QLabel('建模中,结果正在路上...')
        self.label.setAlignment(Qt.AlignHCenter)
        vlayout.addWidget(self.label)
        self.setLayout(vlayout)

class ModeResWidget(QWidget):
    def __init__(self):
        super(ModeResWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 800)
        self.vlayout = QVBoxLayout()
        self.runing_status = RunStatusWidget()
        self.vlayout.addWidget(self.runing_status)
        self.setLayout(self.vlayout)

    def hideRuningBar(self):
        self.runing_status.setVisible(False)

    def addTableView(self,*args):
        self.vlayout.addWidget(TableViewWidget(*args))

    def addTextEit(self):
        self.vlayout.addWidget(QTextEdit())








if __name__=='__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = ModeResWidget()
    w.show()
    sys.exit(app.exec_())