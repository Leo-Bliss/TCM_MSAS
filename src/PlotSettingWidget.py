#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/30 0030 11:58
#@Author  :    tb_youth
#@FileName:    PlotSettingWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth
import sys

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtWidgets import QGroupBox, QLabel, QComboBox, QSpinBox, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout, QLineEdit, QStatusBar

from src.BasicModules import CheckboxEdit,CheckboxSpinBox,CheckboxComBox


class PlotSettingWidget(QWidget):
    def __init__(self):
        super(PlotSettingWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 800)
        plot_type_list = ['折线图', '散点图', '箱线图', '柱状图']
        color_list = ['r', 'g', 'b', 'y', 'k', 'w']
        line_type_list = ['--', '-', '-.', ':']
        mark_type_list = ['*', 'o', '.', '+', 'x', '^', 'v', '<', '>']

        # type
        self.plot_type_label = QLabel('类型选择')
        self.plot_type_combox = QComboBox()
        self.plot_type_combox.addItems(plot_type_list)
        self.show_example = QCheckBox('显示图例')
        grid_layout1 = QGridLayout()
        grid_layout1.addWidget(self.plot_type_label, 0, 0)
        grid_layout1.addWidget(self.plot_type_combox, 0, 1)
        self.plot_type_groupbox = QGroupBox('图形')
        self.plot_type_groupbox.setLayout(grid_layout1)

        # 一般设置
        self.line_type_label = QLabel('线条样式')
        self.line_type_combox = QComboBox()
        self.line_type_combox.addItems(line_type_list)
        self.line_type_combox.setEditable(True)
        self.line_color_lable = QLabel('线条颜色')
        self.line_color_combox = QComboBox()
        self.line_color_combox.addItems(color_list)
        self.line_color_combox.setEditable(True)
        self.line_color_combox.setCurrentIndex(2)
        self.line_width_lable = QLabel('线条宽度')
        self.line_width_spinbox = QSpinBox()
        self.line_width_spinbox.setRange(1, 100)
        ##marker
        self.mark_type_label = QLabel('标记样式')
        self.mark_type_combox = QComboBox()
        self.mark_type_combox.addItems(mark_type_list)
        self.mark_type_combox.setEditable(True)
        self.mark_color_lable = QLabel('标记颜色')
        self.mark_color_combox = QComboBox()
        self.mark_color_combox.addItems(color_list)
        self.mark_color_combox.setEditable(True)

        grid_layout2 = QGridLayout()
        grid_layout2.addWidget(self.line_type_label, 0, 0)
        grid_layout2.addWidget(self.line_type_combox, 0, 1)
        grid_layout2.addWidget(self.line_color_lable, 0, 2)
        grid_layout2.addWidget(self.line_color_combox, 0, 3)
        grid_layout2.addWidget(self.line_width_lable, 1, 0)
        grid_layout2.addWidget(self.line_width_spinbox, 1, 1)
        grid_layout2.addWidget(self.mark_type_label, 1, 2)
        grid_layout2.addWidget(self.mark_type_combox, 1, 3)
        grid_layout2.addWidget(self.mark_color_lable, 2, 0)
        grid_layout2.addWidget(self.mark_color_combox, 2, 1)
        grid_layout2.setHorizontalSpacing(25)

        self.general_groupbox = QGroupBox('一般设置')
        self.general_groupbox.setLayout(grid_layout2)

        # 通用设置
        ##title
        self.title_checkbox_edit = CheckboxEdit('设置图形标题')
        self.title_groupbox = QGroupBox('标题（选填）')
        self.title_groupbox.setLayout(self.title_checkbox_edit)

        ##label
        self.draw_label_checkbox_edit = CheckboxEdit('设置图形标签')
        self.label_groupbox = QGroupBox('标签（选填）')
        self.label_groupbox.setLayout(self.draw_label_checkbox_edit)

        ##ylim
        self.y_max_checkbox_spinbox = CheckboxSpinBox('y轴最大值')
        self.y_min_checkbox_spinbox = CheckboxSpinBox('y轴最小值')
        self.show_x_grid = QCheckBox('显示垂直网格线')
        ##xlim
        self.x_max_checkbox_spinbox = CheckboxSpinBox('x轴最大值')
        self.x_min_checkbox_spinbox = CheckboxSpinBox('x轴最小值')
        self.show_y_grid = QCheckBox('显示水平网格线')

        grid_layout3 = QGridLayout()
        grid_layout3.addItem(self.y_min_checkbox_spinbox, 0, 0, 1, 2)
        grid_layout3.addItem(self.y_max_checkbox_spinbox, 0, 2, 1, 2)
        grid_layout3.addItem(self.x_min_checkbox_spinbox, 1, 0, 1, 2)
        grid_layout3.addItem(self.x_max_checkbox_spinbox, 1, 2, 1, 2)

        ##x,y轴标签
        self.x_lable_checkbox_edit = CheckboxEdit('x轴标签 ')
        self.y_lable_checkbox_edit = CheckboxEdit('y轴标签 ')

        grid_layout3.addItem(self.x_lable_checkbox_edit, 2, 0, 1, 2)
        grid_layout3.addItem(self.y_lable_checkbox_edit, 2, 2, 1, 2)
        grid_layout3.addWidget(self.show_y_grid, 3, 0)
        grid_layout3.addWidget(self.show_x_grid, 3, 2)
        grid_layout3.setSpacing(10)

        self.axis_groupbox = QGroupBox('坐标轴设置（选填）')
        self.axis_groupbox.setLayout(grid_layout3)

        ##标上数值
        self.point_distance_checkbox_spinbox = CheckboxSpinBox('上下距离')
        left, right, = -2147483647, 2147483647
        self.point_distance_checkbox_spinbox.setRange(left, right)
        self.data_groupbox = QGroupBox('标上数值（选填）')
        self.data_groupbox.setLayout(self.point_distance_checkbox_spinbox)

        ##其他设置：xtick,ytick的颜色和旋转角度，网格线颜色，样式等
        self.xticks_color_checkbox_combox = CheckboxComBox('xticks颜色')
        self.xticks_color_checkbox_combox.setItems(color_list)
        self.xticks_color_checkbox_combox.setDefualtValue(4)
        self.xticks_rotation_checkbox_spinbox = CheckboxSpinBox('xticks旋转角度')
        self.xticks_rotation_checkbox_spinbox.setRange(0, 360)

        self.yticks_color_checkbox_combox = CheckboxComBox('yticks颜色')
        self.yticks_color_checkbox_combox.setItems(color_list)
        self.yticks_color_checkbox_combox.setDefualtValue(4)
        self.yticks_rotation_checkbox_spinbox = CheckboxSpinBox('yticks旋转角度')
        self.yticks_rotation_checkbox_spinbox.setRange(0, 360)

        self.gridline_color_checkbox_combox = CheckboxComBox('网格线颜色')
        self.gridline_color_checkbox_combox.setItems(color_list)
        self.gridline_color_checkbox_combox.setDefualtValue(4)
        self.gridline_style_checkbox_combox = CheckboxComBox('网格线样式')
        self.gridline_style_checkbox_combox.setItems(line_type_list)
        # self.gridline_style_checkbox_combox.setDefualtValue(1)

        grid_layout4 = QGridLayout()
        grid_layout4.addItem(self.xticks_color_checkbox_combox, 0, 0, 1, 2)
        grid_layout4.addItem(self.xticks_rotation_checkbox_spinbox, 0, 2, 1, 2)
        grid_layout4.addItem(self.yticks_color_checkbox_combox, 1, 0, 1, 2)
        grid_layout4.addItem(self.yticks_rotation_checkbox_spinbox, 1, 2, 1, 2)
        grid_layout4.addItem(self.gridline_color_checkbox_combox, 2, 0, 1, 2)
        grid_layout4.addItem(self.gridline_style_checkbox_combox, 2, 2, 1, 2)

        self.other_groupbox = QGroupBox()
        self.other_groupbox.setLayout(grid_layout4)
        self.other_groupbox.hide()
        self.other_checkbox = QCheckBox("展开其他选项（选填）")

        layout = QVBoxLayout()
        layout.addWidget(self.plot_type_groupbox)
        layout.addWidget(self.general_groupbox)
        layout.addWidget(self.title_groupbox)
        layout.addWidget(self.label_groupbox)
        layout.addWidget(self.axis_groupbox)
        layout.addWidget(self.data_groupbox)
        layout.addWidget(self.other_checkbox)
        layout.addWidget(self.other_groupbox)

        self.setLayout(layout)
        # 关联信号和槽
        self.other_checkbox.stateChanged.connect(self.showOther)

    # 展开其他选项
    def showOther(self):
        if self.other_checkbox.isChecked():
            self.other_groupbox.show()
        else:
            self.other_groupbox.hide()

    # 获取用户设置的参数
    def getParameters(self):
        # 一般的参数
        general_parameters_dict = {
            'plot_type': self.plot_type_combox.currentText(),
            'plot_line_style': self.line_type_combox.currentText(),
            'plot_line_color': self.line_color_combox.currentText(),
            'plot_line_width': self.line_width_spinbox.value(),
            'plot_line_marker': self.mark_type_combox.currentText(),
            'plot_marker_color': self.mark_color_combox.currentText()
        }
        # 其他可选参数
        other_parameters_dict = {
            'title': self.title_checkbox_edit.getValue(),
            'lable': self.draw_label_checkbox_edit.getValue(),
            'ylim_max': self.y_max_checkbox_spinbox.getValue(),
            'ylim_min': self.y_min_checkbox_spinbox.getValue(),
            'xlim_max': self.x_max_checkbox_spinbox.getValue(),
            'xlim_min': self.x_min_checkbox_spinbox.getValue(),
            'xlabel': self.x_lable_checkbox_edit.getValue(),
            'ylabel': self.y_lable_checkbox_edit.getValue(),
            'show_x_gridline': self.show_x_grid.isChecked(),
            'show_y_gridline': self.show_y_grid.isChecked(),
            'gridline_color': self.gridline_color_checkbox_combox.getValue(),
            'gridline_style': self.gridline_style_checkbox_combox.getValue(),
            'xticks_color': self.xticks_color_checkbox_combox.getValue(),
            'xticks_rotation': self.xticks_rotation_checkbox_spinbox.getValue(),
            'yticks_color': self.yticks_color_checkbox_combox.getValue(),
            'yticks_rotation': self.yticks_rotation_checkbox_spinbox.getValue(),
            'point_distance': self.point_distance_checkbox_spinbox.getValue()
        }

        return general_parameters_dict, other_parameters_dict



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotSettingWidget()
    window.show()
    sys.exit(app.exec_())