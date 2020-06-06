#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/4/30 0030 11:58
#@Author  :    tb_youth
#@FileName:    PlotSettingWidget.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QDoubleSpinBox
from PyQt5.QtWidgets import QGroupBox, QLabel, QComboBox, QSpinBox, QCheckBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout

from src.BasicModules import CheckboxEdit,CheckboxSpinBox,CheckboxComBox,LineEditButton


class PlotSettingWidget(QWidget):
    def __init__(self):
        super(PlotSettingWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(800, 800)
        plot_type_list = ['折线图', '散点图', '柱状图', '箱线图']
        color_list = ['r', 'g', 'b', 'y', 'k', 'w']
        line_type_list = ['--', '-', '-.', ':']
        mark_type_list = ['*', '.', '+','o', 'x', '^', 'v', '<', '>']

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
        self.line_color_linEdit = LineEditButton('#0000ff')

        self.line_width_lable = QLabel('线条宽度')
        self.line_width_spinbox = QSpinBox()
        self.line_width_spinbox.setRange(1, 100)
        ##marker
        self.mark_type_label = QLabel('标记样式')
        self.mark_type_combox = QComboBox()
        self.mark_type_combox.addItems(mark_type_list)
        self.mark_type_combox.setEditable(True)
        self.mark_color_lable = QLabel('标记颜色')
        self.mark_color_lineEdit = LineEditButton('#ff0000')

        self.color_alpha_label = QLabel('图形透明度')
        self.color_alpha_spinbox = QDoubleSpinBox()
        self.color_alpha_spinbox.setDecimals(2)
        self.color_alpha_spinbox.setValue(1)
        self.color_alpha_spinbox.setRange(0,1)
        self.color_alpha_spinbox.setSingleStep(0.01)

        self.grid_layout2 = QGridLayout()
        self.grid_layout2.addWidget(self.line_type_label, 0, 0)
        self.grid_layout2.addWidget(self.line_type_combox, 0, 1)
        self.grid_layout2.addWidget(self.line_color_lable, 0, 2)
        self.grid_layout2.addItem(self.line_color_linEdit, 0, 3)
        self.grid_layout2.addWidget(self.line_width_lable, 1, 0)
        self.grid_layout2.addWidget(self.line_width_spinbox, 1, 1)
        self.grid_layout2.addWidget(self.mark_type_label, 1, 2)
        self.grid_layout2.addWidget(self.mark_type_combox, 1, 3)
        self.grid_layout2.addWidget(self.mark_color_lable, 2, 0)
        self.grid_layout2.addItem(self.mark_color_lineEdit, 2, 1)
        self.grid_layout2.addWidget(self.color_alpha_label, 2, 2)
        self.grid_layout2.addWidget(self.color_alpha_spinbox, 2, 3)
        self.grid_layout2.setHorizontalSpacing(25)

        self.general_groupbox = QGroupBox('一般设置')
        self.general_groupbox.setLayout(self.grid_layout2)

        # 通用设置
        ##title
        self.title_checkbox_edit = CheckboxEdit('设置图形标题','绘图示例效果图')
        self.title_checkbox_edit.checkbox.setChecked(True)
        self.title_groupbox = QGroupBox('标题（可选）')
        self.title_groupbox.setLayout(self.title_checkbox_edit)

        ##label
        self.draw_label_checkbox_edit = CheckboxEdit('设置图形标签','示例')
        self.draw_label_checkbox_edit.checkbox.setChecked(True)

        self.label_groupbox = QGroupBox('标签（可选）')
        self.label_groupbox.setLayout(self.draw_label_checkbox_edit)

        ##ylim
        self.y_lim_checkbox_edit = CheckboxEdit('y轴范围')
        self.y_lim_checkbox_edit.setHolderText('min,max')
        ##xlim
        self.x_lim_checkbox_edit = CheckboxEdit('x轴范围')
        self.x_lim_checkbox_edit.setHolderText('min,max')
        self.show_x_grid = QCheckBox('显示垂直网格线')
        self.show_y_grid = QCheckBox('显示水平网格线')

        grid_layout3 = QGridLayout()
        grid_layout3.addItem(self.y_lim_checkbox_edit, 0, 0, 1, 2)
        grid_layout3.addItem(self.x_lim_checkbox_edit, 0, 2, 1, 2)

        ##x,y轴标签
        self.x_lable_checkbox_edit = CheckboxEdit('x轴标签 ','x')
        self.x_lable_checkbox_edit.checkbox.setChecked(True)
        self.y_lable_checkbox_edit = CheckboxEdit('y轴标签 ','y')
        self.y_lable_checkbox_edit.checkbox.setChecked(True)


        grid_layout3.addItem(self.x_lable_checkbox_edit, 2, 0, 1, 2)
        grid_layout3.addItem(self.y_lable_checkbox_edit, 2, 2, 1, 2)
        grid_layout3.addWidget(self.show_y_grid, 3, 0)
        grid_layout3.addWidget(self.show_x_grid, 3, 2)
        grid_layout3.setSpacing(10)

        self.axis_groupbox = QGroupBox('坐标轴设置（可选）')
        self.axis_groupbox.setLayout(grid_layout3)

        ##标上数值 :偏移量可能是float，int,简化处理使用lineEdit
        ### y的偏移量
        self.point_ydistance_checkbox_edit = CheckboxEdit('垂直偏移(y)')
        ### x的偏移量
        self.point_xdistance_checkbox_edit = CheckboxEdit('水平偏移(x)')
        hlayout = QHBoxLayout()
        hlayout.addItem(self.point_ydistance_checkbox_edit)
        hlayout.addItem(self.point_xdistance_checkbox_edit)
        self.data_groupbox = QGroupBox('标上数值（可选）')
        self.data_groupbox.setLayout(hlayout)


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

        grid_layout4 = QGridLayout()
        grid_layout4.addItem(self.xticks_color_checkbox_combox, 0, 0, 1, 2)
        grid_layout4.addItem(self.xticks_rotation_checkbox_spinbox, 0, 2, 1, 2)
        grid_layout4.addItem(self.yticks_color_checkbox_combox, 1, 0, 1, 2)
        grid_layout4.addItem(self.yticks_rotation_checkbox_spinbox, 1, 2, 1, 2)
        grid_layout4.addItem(self.gridline_color_checkbox_combox, 2, 0, 1, 2)
        grid_layout4.addItem(self.gridline_style_checkbox_combox, 2, 2, 1, 2)

        self.more_groupbox = QGroupBox()
        self.more_groupbox.setLayout(grid_layout4)
        self.more_groupbox.hide()
        self.more_checkbox = QCheckBox("展开更多选项（可选）")

        layout = QVBoxLayout()
        layout.addWidget(self.plot_type_groupbox)
        layout.addWidget(self.general_groupbox)
        layout.addWidget(self.title_groupbox)
        layout.addWidget(self.label_groupbox)
        layout.addWidget(self.axis_groupbox)
        layout.addWidget(self.data_groupbox)
        layout.addWidget(self.more_checkbox)
        layout.addWidget(self.more_groupbox)

        self.setLayout(layout)
        # 关联信号和槽
        self.more_checkbox.stateChanged.connect(self.showMore)
        self.plot_type_combox.currentIndexChanged.connect(self.indexChanged)

    def indexChanged(self,index):
        if index == 0:
            self.lineGenneral()
        elif index == 1:
            self.scatterGeneral()
        elif index == 2:
            self.columnBarGeneral()

    def scatterGeneral(self):
        self.line_type_label.setVisible(False)
        self.line_type_combox.setVisible(False)
        self.line_color_lable.setVisible(False)
        self.line_color_linEdit.hideWidget()
        self.line_width_lable.setText('散点大小')
        self.line_width_spinbox.setValue(20)
        self.mark_type_label.setText('散点样式')
        pass

    def columnBarGeneral(self):
        self.line_type_label.setVisible(False)
        self.line_type_combox.setVisible(False)
        self.line_color_lable.setVisible(False)
        self.line_color_linEdit.hideWidget()
        self.mark_type_label.setVisible(False)
        self.mark_type_combox.setVisible(False)
        self.line_width_spinbox.setVisible(False)
        self.line_width_lable.setText('柱形宽度')
        self.mark_color_lable.setText('柱形颜色')
        self.bar_width_spinbox = QDoubleSpinBox()
        self.bar_width_spinbox.setDecimals(2)
        self.bar_width_spinbox.setValue(0.20)
        self.bar_width_spinbox.setRange(0, 1)
        self.bar_width_spinbox.setSingleStep(0.01)
        self.grid_layout2.addWidget(self.bar_width_spinbox, 1, 1)



    def lineGenneral(self):
        self.line_type_label.setVisible(True)
        self.line_type_combox.setVisible(True)
        self.line_color_lable.setVisible(True)
        self.line_width_lable.setText('线条宽度')
        self.line_width_spinbox.setValue(1)
        self.mark_type_label.setText('标记样式')
        self.mark_type_label.setVisible(True)
        self.mark_type_combox.setVisible(True)
        self.line_color_lable.setVisible(True)
        self.line_color_linEdit.showWidget()
        pass


    def initGeneralSetting(self):
        pass

    
    
    # 展开其他选项
    def showMore(self):
        self.more_groupbox.setVisible(self.more_checkbox.isChecked())

    # 一般的参数
    def getGeneralParameters(self):
        ##折线图,散点图(只用后四个参数)
        general_parameters_dict = {
            'plot_type': self.plot_type_combox.currentIndex(),
            'plot_line_style': self.line_type_combox.currentText(),
            'plot_line_color': self.line_color_linEdit.getText(),
            'plot_line_width': self.line_width_spinbox.value(),
            'plot_line_marker': self.mark_type_combox.currentText(),
            'plot_marker_color': self.mark_color_lineEdit.getText(),
            'plot_marker_color_alpha':self.color_alpha_spinbox.value()
        }
        # 柱状图
        if hasattr(self,'bar_width_spinbox'):
            general_parameters_dict['bar_width'] = self.bar_width_spinbox.value()

        return general_parameters_dict

    # 其他可选参数，通用的参数
    def getOtherParameters(self):
        other_parameters_dict = {
            'show_x_gridline': self.show_x_grid.isChecked(),
            'show_y_gridline': self.show_y_grid.isChecked(),
            'gridline_color': self.gridline_color_checkbox_combox.getValue(),
            'gridline_style': self.gridline_style_checkbox_combox.getValue(),
            'xticks_color': self.xticks_color_checkbox_combox.getValue(),
            'xticks_rotation': self.xticks_rotation_checkbox_spinbox.getValue(),
            'yticks_color': self.yticks_color_checkbox_combox.getValue(),
            'yticks_rotation': self.yticks_rotation_checkbox_spinbox.getValue()
        }

        # 数值偏移量
        if self.point_xdistance_checkbox_edit.isChecked() or self.point_ydistance_checkbox_edit.isChecked():
            step_x, step_y = 0, 0
            if self.point_xdistance_checkbox_edit.isChecked():
                step_x = self.point_xdistance_checkbox_edit.getValue()
            if self.point_ydistance_checkbox_edit.isChecked():
                step_y  = self.point_ydistance_checkbox_edit.getValue()
            if step_x == '':
                step_x = 0
            if step_y == '':
                step_y = 0
            other_parameters_dict['point_distance'] = (step_y, step_x)

        if self.title_checkbox_edit.isChecked():
            other_parameters_dict['title'] = self.title_checkbox_edit.getValue()
        if self.draw_label_checkbox_edit.isChecked():
            other_parameters_dict['lable'] = self.draw_label_checkbox_edit.getValue()
        if self.x_lable_checkbox_edit.isChecked():
            other_parameters_dict['xlabel'] = self.x_lable_checkbox_edit.getValue()
        if self.y_lable_checkbox_edit.isChecked():
            other_parameters_dict['ylabel'] = self.y_lable_checkbox_edit.getValue()

        # tlim,xlim
        if self.y_lim_checkbox_edit.isChecked():
            other_parameters_dict['ylim'] = self.y_lim_checkbox_edit.getRange()
        if self.x_lim_checkbox_edit.isChecked():
            other_parameters_dict['xlim'] = self.x_lim_checkbox_edit.getRange()
        return other_parameters_dict
        

    # 获取用户设置的参数
    def getParameters(self):
        return self.getGeneralParameters(),self.getOtherParameters()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlotSettingWidget()
    window.show()
    sys.exit(app.exec_())