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
from PyQt5.QtCore import QSize, pyqtProperty, QTimer, Qt
from PyQt5.QtGui import QColor, QPainter
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

class CircleProgressBar(QWidget):

    Color = QColor(24, 189, 155)  # 圆圈颜色
    Clockwise = True  # 顺时针还是逆时针
    Delta = 36

    def __init__(self, *args, color=None, clockwise=True, **kwargs):
        super(CircleProgressBar, self).__init__(*args, **kwargs)
        self.angle = 0
        self.Clockwise = clockwise
        if color:
            self.Color = color
        self._timer = QTimer(self, timeout=self.update)
        self._timer.start(100)

    def paintEvent(self, event):
        super(CircleProgressBar, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        side = min(self.width(), self.height())
        painter.scale(side / 100.0, side / 100.0)
        painter.rotate(self.angle)
        painter.save()
        painter.setPen(Qt.NoPen)
        color = self.Color.toRgb()
        for i in range(11):
            color.setAlphaF(1.0 * i / 10)
            painter.setBrush(color)
            painter.drawEllipse(10, -15, 8, 8)
            painter.rotate(36)
        painter.restore()
        self.angle += self.Delta if self.Clockwise else -self.Delta
        self.angle %= 360

    @pyqtProperty(QColor)
    def color(self) -> QColor:
        return self.Color

    @color.setter
    def color(self, color: QColor):
        if self.Color != color:
            self.Color = color
            self.update()

    @pyqtProperty(bool)
    def clockwise(self) -> bool:
        return self.Clockwise

    @clockwise.setter
    def clockwise(self, clockwise: bool):
        if self.Clockwise != clockwise:
            self.Clockwise = clockwise
            self.update()

    @pyqtProperty(int)
    def delta(self) -> int:
        return self.Delta

    @delta.setter
    def delta(self, delta: int):
        if self.delta != delta:
            self.delta = delta
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(100, 100)


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
        '''
        由于导出接口的原因，暂时只支持xlsx格式导出；
        否则容易出现乱码问题。
        ;;csv(*.csv);;xls(*.xls)这两种到时再处理
        '''
        file_path, _ = QFileDialog.getSaveFileName(self, '保存文件', '../data',
                                                   'xlsx(*.xlsx)')
        if file_path == '':
            return
        # 文件中写入数据
        try:
            self.write_thread = WriteExcelThread(file_path, self.data_list)
            self.write_thread.start_signal.connect(self.showStatus)
            self.write_thread.end_signal.connect(self.write_thread.quit)
            self.write_thread.start()
        except Exception as e:
            print(e)


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