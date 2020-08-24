#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/11 0011 23:56
#@Author  :    tb_youth
#@FileName:    childFatherWindow.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth


from PyQt5.QtWidgets import QWidget, QDialog, QApplication, QVBoxLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import Qt

class ChildWidget(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.resize(500,500)
        self.setWindowTitle('Child')
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowMinimizeButtonHint)


class ParentWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(600,600)
        self.setWindowTitle('Parent')
        self.btn = QPushButton('show child')
        self.child_widget = ChildWidget(self)
        layout = QVBoxLayout()
        layout.addWidget(self.btn)
        self.setLayout(layout)
        self.btn.clicked.connect(self.onClickedBtn)

    def onClickedBtn(self):
        self.child_widget.exec_()





if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    w = ParentWidget()
    w.show()
    sys.exit(app.exec_())
