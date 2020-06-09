#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/1 0001 12:06
#@Author  :    tb_youth
#@FileName:    ProgressBar.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
进度条
'''


from PyQt5.QtCore import QSize, pyqtProperty, QTimer, Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget


class CircleProgressBar(QWidget):
    # 圆圈颜色
    Color = QColor(24, 189, 155)
    # 顺时针
    Clockwise = True
    # 角度步长，可以调节转速
    Delta = 36

    def __init__(self, *args, color=None, clockwise=True, **kwargs):
        super(CircleProgressBar, self).__init__(*args, **kwargs)
        self.setFixedSize(30,30)
        self.angle = 0
        self.Clockwise = clockwise
        if color:
            self.Color = color
        '''
        实例化Qtimer，连接timeout()信号到适当的槽函数，并调用start()，
        然后在恒定的时间间隔会发射timeout()信号。
        '''
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
            painter.drawEllipse(30, -10, 20, 20)
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


if __name__ =='__main__':
    pass