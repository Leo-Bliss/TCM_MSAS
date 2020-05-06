#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :    2020/4/30 0030 13:03
# @Author  :    tb_youth
# @FileName:    PlotWidget.py
# @SoftWare:    PyCharm
# @Blog    :    https://blog.csdn.net/tb_youth


from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton
import matplotlib.pyplot as wd_plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar


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

        self.wd_plt = wd_plt
        # 中文乱码处理
        self.wd_plt.rcParams['font.sans-serif'] = ['SimHei']
        self.wd_plt.rcParams['axes.unicode_minus'] = False

        self.plot_type = 0
        self.y_list, self.x_list = [], []
        self.general_parameters_dict = {}
        self.other_parameters_dict = {}
        # 创建一个展示板
        self.figure = self.wd_plt.figure(facecolor='w')


        # 使用pyplot里面的figure 显示窗口的时候会出现短暂的黑屏，待解决。。（暂时没有找到解决方案）
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.continue_draw_btn)
        hlayout.addWidget(self.reDraw_btn)
        hlayout.addStretch()
        hlayout.setSpacing(20)

        vlayout = QVBoxLayout()
        vlayout.addItem(hlayout)
        vlayout.addWidget(self.canvas)
        vlayout.addWidget(self.toolbar)
        vlayout.setStretch(0, 2)
        vlayout.setStretch(1, 9)
        vlayout.setStretch(2, 1)
        vlayout.setSpacing(5)

        self.setLayout(vlayout)
        self.showDrawExample()
        self.reDraw_btn.clicked.connect(self.onClickedReDrawButton)
        self.continue_draw_btn.clicked.connect(self.onClickedContinueDrawButton)

    def showDrawExample(self):
        y_list, x_list = [1.22, 2.43, 5.78, 9.65, 7.24], ['a','b','c','d','e']
        general_parameters_dict = {
            'plot_line_style': '--',
            'plot_line_color': 'b',
            'plot_line_width': 1,
            'plot_line_marker': '*',
            'plot_marker_color': 'r',
            'plot_marker_color_alpha': 1
        }
        other_parameters_dict = {
            'title':'绘图示例',
            'lable':'示例1',
            'xlabel':'x',
            'ylabel':'y',
            'point_distance':(0.15,0)
        }
        self.setPlotData(y_list,x_list,general_parameters_dict,other_parameters_dict)
        self.onClickedContinueDrawButton()

    def setPlotData(self, y_list, x_list, general_parameters_dict, other_parameters_dict):
        self.y_list, self.x_list = y_list, x_list
        self.general_parameters_dict, self.other_parameters_dict = general_parameters_dict, other_parameters_dict

    def onClickedContinueDrawButton(self):
        self.draw()

    def onClickedReDrawButton(self):
        self.figure.clear()
        self.canvas.draw()
        self.draw()



    def judge_num(self, num):
        try:
            float(num)
            return True
        except:
            return False

    # 折线图
    def drawLine(self):
        self.wd_plt.plot(self.x_list, self.y_list, linestyle=self.general_parameters_dict.get('plot_line_style', '--'),
                         linewidth=self.general_parameters_dict.get('plot_line_width'),
                         color=self.general_parameters_dict.get('plot_line_color', 'r'),
                         marker=self.general_parameters_dict.get('plot_line_marker', '*'),
                         mec=self.general_parameters_dict.get('plot_marker_color', 'r'),
                         label=self.other_parameters_dict.get('lable'),
                         alpha=self.general_parameters_dict.get('plot_marker_color_alpha', 1.00))

    # 散点图
    def drawScatter(self):
        self.wd_plt.scatter(self.x_list, self.y_list,
                            c=self.general_parameters_dict.get('plot_marker_color', 'r'),
                            s=self.general_parameters_dict.get('plot_line_width'),
                            alpha=self.general_parameters_dict.get('plot_marker_color_alpha', 20.00),
                            marker=self.general_parameters_dict.get('plot_line_marker', 'o'),
                            label=self.other_parameters_dict.get('lable'))

    def draw(self):
        try:
            # 这里开始按照matlibplot的方式绘图
            # 垂直网格线
            if self.other_parameters_dict.get('show_y_gridline', None):
                self.wd_plt.grid(axis='y', color=self.other_parameters_dict.get('gridline_color', 'r'),
                                 linestyle=self.other_parameters_dict.get('gridline_style', '-'), linewidth=1)

            # 水平网格项
            if self.other_parameters_dict.get('show_x_gridline', None):
                self.wd_plt.grid(axis='x', color=self.other_parameters_dict.get('gridline_color', 'r'),
                                 linestyle=self.other_parameters_dict.get('gridline_style'), linewidth=1)
            # init_xticks
            if len(self.x_list) != len(self.y_list):
                self.x_list = list(range(1, len(self.y_list) + 1))

            # xlim,ylim
            y_lim = self.other_parameters_dict.get('ylim', None)
            if y_lim:
                self.wd_plt.ylim(y_lim)
            x_lim = self.other_parameters_dict.get('xlim', None)
            if x_lim:
                self.wd_plt.xlim(x_lim)
            self.plot_type = self.general_parameters_dict.get('plot_type',0)
            if self.plot_type == 0:
                self.drawLine()
            elif self.plot_type == 1:
                self.drawScatter()

            # 调整ticks颜色，角度
            xticks_color = self.other_parameters_dict.get('xticks_color', 'k')
            xticks_rotation = self.other_parameters_dict.get('xticks_rotation',0)
            yticks_color = self.other_parameters_dict.get('yticks_color', 'k')
            yticks_rotation = self.other_parameters_dict.get('yticks_rotation',0)
            self.wd_plt.xticks(color=xticks_color, rotation=xticks_rotation)
            self.wd_plt.yticks(color=yticks_color, rotation=yticks_rotation)

            # 图形label显示
            if self.other_parameters_dict.get('lable', None):
                self.wd_plt.legend()

            # 标上数值
            step_y, step_x = self.other_parameters_dict.get('point_distance')
            if self.judge_num(step_y):
                if not self.judge_num(step_x):
                    step_x = 0
                if not isinstance(self.x_list[0],str):
                    for x, y in zip(self.x_list, self.y_list):
                        self.wd_plt.text(x+step_x, y + step_y, '%s' % y, ha='center')
                else:
                    for x, y in zip(self.x_list, self.y_list):
                        self.wd_plt.text(x, y + step_y, '%s' % y, ha='center')

            # 设置标题
            tilte = self.other_parameters_dict.get('title', None)
            if tilte:
                self.wd_plt.title(tilte, fontsize=12)

            # x,y轴label
            x_label = self.other_parameters_dict.get('xlabel', None)
            y_label = self.other_parameters_dict.get('ylabel', None)
            if x_label:
                self.wd_plt.xlabel(x_label, fontsize=15)
            if y_label:
                self.wd_plt.ylabel(y_label, fontsize=15)

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
