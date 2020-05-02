#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/30 0030 13:03
#@Author  :    tb_youth
#@FileName:    PlotWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth



from PyQt5.QtWidgets import QWidget,QFileDialog
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QStatusBar
import matplotlib.pyplot as wd_plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.MyThreads import SaveImgThread

'''
绘制图形界面
'''


class PlotWidget(QWidget):
    def __init__(self):
        super(PlotWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(1200, 800)
        self.setWindowTitle('建模分析')
        self.continue_draw_btn = QPushButton("继续绘制")
        self.continue_draw_btn.setToolTip('将在现在图形基础上继续绘制~')
        self.reDraw_btn = QPushButton("重新绘制")
        self.reDraw_btn.setToolTip('将清除原有图形重新绘制~')
        self.output_btn = QPushButton("导出")

        self.wd_plt = wd_plt
        self.y_list, self.x_list = [],[]
        self.general_parameters_dict, self.other_parameters_dict = {},{}
        # 创建一个展示板
        self.figure = self.wd_plt.figure(facecolor='w')
        # 中文乱码处理
        self.wd_plt.rcParams['font.sans-serif'] = ['SimHei']
        self.wd_plt.rcParams['axes.unicode_minus'] = False

        #使用pyplot里面的figure 显示窗口的时候会出现短暂的黑屏，待解决。。（暂时没有找到解决方案）
        self.canvas = FigureCanvas(self.figure)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.continue_draw_btn)
        hlayout.addWidget(self.reDraw_btn)
        hlayout.addWidget(self.output_btn)
        hlayout.addStretch()
        hlayout.setSpacing(20)
        self.status_bar = QStatusBar()
        #self.status_bar.showMessage('状态栏', 5000)

        vlayout = QVBoxLayout()
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(self.status_bar)
        vlayout.setStretch(0, 2)
        vlayout.setStretch(1, 9)
        vlayout.setStretch(2, 1)
        vlayout.setSpacing(5)

        self.setLayout(vlayout)

        self.reDraw_btn.clicked.connect(self.onClickedReDrawButton)
        self.continue_draw_btn.clicked.connect(self.onClickedContinueDrawButton)
        self.output_btn.clicked.connect(self.onClickedSave)

    def setPlotData(self,y_list,x_list,general_parameters_dict,other_parameters_dict):
        self.y_list,self.x_list = y_list,x_list
        self.general_parameters_dict,self.other_parameters_dict = general_parameters_dict,other_parameters_dict

    def onClickedContinueDrawButton(self):
        self.draw()

    def onClickedReDrawButton(self):
        self.figure.clear()
        self.canvas.draw()
        self.draw()

    def onClickedSave(self):
        self.status_bar.showMessage('保存文件', 5000)
        file_path, _ = QFileDialog.getSaveFileName(self, '保存文件', '../data',
                                                   'svg(*.svg);;png(*.png)')
        if file_path == '':
            return
        # 文件中写入数据
        try:
            self.save_img_thread = SaveImgThread(file_path, self.wd_plt)
            self.save_img_thread.start_signal.connect(self.showStatus)
            self.save_img_thread.end_signal.connect(self.save_img_thread.quit)
            self.save_img_thread.start()
            self.status_bar.showMessage('保存完毕！')
        except Exception as e:
            print(e)

    def showStatus(self, msg):
        self.status_bar.showMessage(str(msg), 5000)

    def draw(self):
        try:
            # 这里开始按照matlibplot的方式绘图
            # 垂直网格线
            if self.other_parameters_dict.get('show_y_gridline'):
                self.wd_plt.grid(axis='y', color=self.other_parameters_dict.get('gridline_color','r'),
                              linestyle=self.other_parameters_dict.get('gridline_style','-'), linewidth=1)

            # 水平网格项
            if self.other_parameters_dict.get('show_x_gridline'):
                self.wd_plt.grid(axis='x', color=self.other_parameters_dict.get('gridline_color','r'),
                              linestyle=self.other_parameters_dict.get('gridline_style'), linewidth=1)
            # init_xticks
            x = list(range(1, len(self.y_list) + 1))
            print(x)
            # xlim,ylim
            ylim_max = self.other_parameters_dict.get('ylim_max')
            ylim_min = self.other_parameters_dict.get('ylim_min')
            if ylim_max and ylim_min and ylim_max - ylim_min >= max(self.y_list):
                # print(ylim_min, ylim_max)
                self.wd_plt.ylim(ylim_min, ylim_max)
            xlim_max = self.other_parameters_dict.get('xlim_max')
            xlim_min = self.other_parameters_dict.get('xlim_min')
            if len(self.x_list) == len(self.y_list):
                x = self.x_list
            else:
                if xlim_max:
                    x = list(range(xlim_max - len(self.y_list) + 1, xlim_max + 1))
                if xlim_min:
                    x = list(range(xlim_min, xlim_min + len(self.y_list)))
            # 绘制
            self.wd_plt.plot(x, self.y_list, linestyle=self.general_parameters_dict.get('plot_line_style','--'),
                          linewidth=self.general_parameters_dict.get('plot_line_width'),
                          color=self.general_parameters_dict.get('plot_line_color','b'),
                          marker=self.general_parameters_dict.get('plot_line_marker','*'),
                          mec=self.general_parameters_dict.get('plot_marker_color','r'),
                          label=self.other_parameters_dict.get('lable'))

            # 调整ticks颜色，角度
            xticks_color = self.other_parameters_dict.get('xticks_color','k')
            xticks_rotation = self.other_parameters_dict.get('xticks_rotation')
            yticks_color = self.other_parameters_dict.get('yticks_color','k')
            yticks_rotation = self.other_parameters_dict.get('yticks_rotation')
            self.wd_plt.xticks(color=xticks_color, rotation=xticks_rotation)
            self.wd_plt.yticks(color=yticks_color, rotation=yticks_rotation)

            # 图形label显示
            if self.other_parameters_dict.get('lable'):
                self.wd_plt.legend()

            # 标上数值
            point_distance = self.other_parameters_dict.get('point_distance')
            if point_distance is not None:
                for x, y in enumerate(self.y_list):
                    self.wd_plt.text(x, y + point_distance, '%s' % y, ha='center')

            # 设置标题
            tilte = self.other_parameters_dict.get('title')
            if tilte:
                self.wd_plt.title(tilte, fontsize=10)

            # x,y轴label
            x_label = self.other_parameters_dict.get('xlabel')
            y_label = self.other_parameters_dict.get('ylabel')
            if x_label:
                self.wd_plt.xlabel(x_label, fontsize=10)
            if y_label:
                self.wd_plt.ylabel(y_label, fontsize=10)
            # 按照matlibplot的方式绘制之后，在窗口上绘制
            self.canvas.draw()
        except Exception as e:
            print(e)



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = PlotWidget()
    window.show()
    sys.exit(app.exec_())