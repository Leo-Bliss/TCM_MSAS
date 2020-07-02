#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/6/6 0006 21:44
#@Author  :    tb_youth
#@FileName:    AboutWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
关于本软件
'''


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QTextEdit, QVBoxLayout, QFrame, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QLabel


class AboutWidget(QDialog):
    def __init__(self,parent=None):
        super(AboutWidget,self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.setFixedSize(750,620)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle('关于')
        self.text_edit = QTextEdit()
        self.setStyleSheet('''
         background-color:#5fa7c8;
         font-size:20px;
         font-family:verdana;
         padding:20px;
         
        ''')
        self.setWindowOpacity(0.9)
        self.close_btn = QPushButton('x')
        self.close_btn.setStyleSheet('''
        QPushButton{
        border:none;
        color:white;
        font-size:15px;
        border-radius:2px;
        }
        QPushButton:hover{
        color:red;
        background:white;
        font-weight:bold;
        }
        ''')
        self.close_btn.clicked.connect(self.close)
        about_html = '''
        <!DOCTYPE html>
        <html>
        <meta charset="utf-8" />
        <title>关于系统</title>
        <style>
        .box{
        opacity:0.3;
        background-color:#5fa7c8;
        }

        </style>
        <div class="box">
        <p class='ename'>E-name: Multifunctional Data Analysis System Optimized For Partial Least Squares(MFDAS_PLS)</p>
        <p>Version: 0.3</p>
        <p> Copyright: &copy;Jiangxi University of Traditional Chinese Medicine</p>
        <p>Developer: tbyouth</p>
        <p>Contributor: Team Of Computer Academy In JXUTCM </p>
        <p>Email: tbyouth11@gmail.com</p>
        <p>HomePage: https://jsj.jxutcm.edu.cn</p>
        </div>
        </html>
        '''
        self.text_edit.setHtml(about_html)
        self.text_edit.setFrameShape(QFrame.NoFrame)
        self.text_edit.setReadOnly(True)
        self.logo_label = QLabel()
        self.logo_label.setPixmap(QPixmap('../imgs/logo.ico'))
        self.name_label = QLabel('优化偏最小二乘的多功能数据分析系统')
        self.name_label.setStyleSheet('''
         QLabel{
        color:white;
        font-size:25px;
        font-weight:bold;
        }
        ''')
        hlayout0 = QHBoxLayout()
        hlayout0.addStretch()
        hlayout0.addWidget(self.close_btn)
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.logo_label)
        hlayout.addWidget(self.name_label)
        hlayout.addStretch()
        vlayout = QVBoxLayout()
        vlayout.addItem(hlayout0)
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.text_edit)
        vlayout.setStretch(0,1)
        vlayout.setStretch(1,3)
        vlayout.setStretch(2,14)
        self.setLayout(vlayout)

    def leaveEvent(self, e):
        pass
        # self.close()

    def mousePressEvent(self, e):
        pass
        # if e.button() == Qt.LeftButton:
        #     self.close()




if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    window = AboutWidget()
    window.show()
    sys.exit(app.exec_())
