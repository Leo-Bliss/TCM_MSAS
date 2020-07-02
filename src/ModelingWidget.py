# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :    2020/4/26 0026 13:19
# @Author  :    tb_youth
# @FileName:    ModelingWidget.py
# @SoftWare:    PyCharm
# @Blog    :    https://blog.csdn.net/tb_youth

'''
运行模型界面，
显示模型摘要，
结果数据，绘制图形等
'''

import sys
from PyQt5.QtWidgets import QApplication, QTabWidget, QStyleFactory, QDialog
from PyQt5.QtWidgets import  QHBoxLayout,QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from src.PlotDataWidget import PlotDataWidget
from src.PlotSettingWidget import PlotSettingWidget
from src.ModeResWidget import ModeResWidget
from src.PlotWidget import PlotWidget
from src.BriefWidget import BriefWidget
from src.MyThreads import TimerThread,WorkerThread
from src.DataConverter import DataConverter

class ModelingWidget(QDialog):
    def __init__(self,model,all_dict,id,parent=None):
        super(ModelingWidget,self).__init__(parent)
        self.model = model
        self.all_dict = all_dict
        self.id = id
        self.initUI()

    def initUI(self):
        self.resize(1300,800)
        QApplication.setStyle(QStyleFactory.keys()[2])
        self.setWindowFlags(Qt.Window|Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
        self.setWindowTitle('建模分析')
        self.timer_thread = None
        self.worker_thread = None
        self.show_data_dict = None
        hlayout = QHBoxLayout()
        self.tab_widget = QTabWidget()

        self.brief_widget = BriefWidget()
        self.tab_widget.addTab(self.brief_widget, '模型摘要')

        self.res_widget = ModeResWidget()
        self.tab_widget.addTab(self.res_widget,'模型结果')

        self.plot_data_widget = PlotDataWidget()
        self.tab_widget.addTab(self.plot_data_widget, '绘图数据')

        self.plot_setting_widget = PlotSettingWidget()
        self.tab_widget.addTab(self.plot_setting_widget, '绘图设置')

        self.plot_widget = PlotWidget()
        # 重新绑定信号槽
        self.plot_widget.continue_draw_btn.disconnect()
        self.plot_widget.continue_draw_btn.clicked.connect(self.onClickedCDraw)
        self.plot_widget.reDraw_btn.disconnect()
        self.plot_widget.reDraw_btn.clicked.connect(self.onClickedRDraw)

        hlayout.addWidget(self.tab_widget)
        hlayout.addWidget(self.plot_widget)
        hlayout.setStretch(0,3)
        hlayout.setStretch(1,7)
        self.setLayout(hlayout)
        # 开始运行模型
        self.work()


    #运行相应的模型
    def work(self):
        self.worker_thread = WorkerThread(self.model,self.all_dict,self.id)
        self.worker_thread.start_signal.connect(self.showProgressBar)
        self.worker_thread.res_signal.connect(self.showMainResInfo)
        self.worker_thread.end_signal.connect(self.hideProgressBar)
        self.timer_thread = TimerThread()
        self.timer_thread.start_signal.connect(self.setTimer)
        self.worker_thread.start()
        self.timer_thread.start()
        self.brief_widget.termination_btn.clicked.connect(self.endRun)

    #显示计时器
    def setTimer(self,mgs):
        self.brief_widget.showTimer(mgs)

    def showProgressBar(self):
        alg_dict = {
            0: 'DSA-PLS',
            1: 'LAPLS',
            2: 'RBM-PLS',
            3: 'SEA-PLS',
            4: 'PLS-S-DA',
            5: 'DBN-PLS',
            6: 'Mtree-PLS',
            7: 'RF-PLS',
            8: 'PLSCF',
            9: 'SBMPLS',
            10: 'GRA-PLS'
        }

        # 模型
        self.brief_widget.appendText('模型：{}'.format(alg_dict.get(self.id)))
        # （部分）变量
        self.brief_widget.setStatus('建模中...')

    def showVarInfo(self):
        # 显示设置的变量
        var_dict = self.all_dict.get('var_dict')
        independ_var_list = var_dict.get('independ_var')
        depend_var_list = var_dict.get('depend_var')
        cnt1 = len(independ_var_list)
        cnt2 = len(depend_var_list)
        show_independ = independ_var_list if cnt1 <= 10 else independ_var_list[0:10]
        show_depend = depend_var_list if cnt2 <= 10 else depend_var_list[0:10]
        tmp_x = str(show_independ)[1:-1]
        tmp_x = tmp_x.replace("'",'')
        #为了美观，暂时最多显示10个变量
        if cnt1 > 10:
            tmp_x += '...'
        tmp_y = str(show_depend)[1:-1]
        tmp_y = tmp_y.replace("'",'')
        if cnt2 > 10:
            tmp_y += '...'
        var_info = '自变量：{}\n因变量：{}'.format(tmp_x, tmp_y)
        self.brief_widget.appendText(var_info)

        self.brief_widget.appendText('-'*40)

        # 显示设置的参数
        parameter_dict = self.all_dict.get('parameter_dict')
        parameter_name_dict = self.all_dict.get('parameter_name_dict')
        if parameter_dict.get('q'):
            parameter_dict['q'] = parameter_dict.get('q') * 100
        for name,value in zip(parameter_name_dict.values(),parameter_dict.values()):
            self.brief_widget.appendText(str(name)+str(value))
            #print(name,value)

    def showMainResInfo(self,dct):
        self.brief_widget.appendText('-'*40)
        for k,v in dct.items():
            if k == 'show_data_dict':
                self.show_data_dict = v
                break
            line = '{}：{}'.format(k,v)
            self.brief_widget.appendText(line)
        self.brief_widget.appendText('-'*40)

    def endRun(self):
        if self.worker_thread:
            self.worker_thread.terminate()
        if self.timer_thread:
            self.timer_thread.is_running = False
        self.brief_widget.setStatus('已经终止建模...')
        self.brief_widget.progressbar.close()
        self.brief_widget.termination_btn.setEnabled(False)



    def hideProgressBar(self):
        self.brief_widget.setStatus('建模已完成...')
        self.brief_widget.termination_btn.setVisible(False)
        self.res_widget.hideRuningBar()
        if self.show_data_dict:
            data_converter = DataConverter()
            for key,value in self.show_data_dict.items():
                data_list = data_converter.DataFrame_to_list2(value)
                self.res_widget.addTableView(key,data_list)
        self.showVarInfo()
        self.brief_widget.appendText('-'*40)
        self.brief_widget.appendText('总耗时：{}s\n'.format(self.brief_widget.getRunTime()))
        if self.worker_thread:
            self.worker_thread.quit()
        if self.timer_thread:
            self.timer_thread.is_running = False
        self.brief_widget.progressbar.close()

    #在原来的基础上继续画图
    def onClickedCDraw(self):
        y_list, x_list = self.plot_data_widget.getData()
        general_parameters_dict, other_parameters_dict = self.plot_setting_widget.getParameters()
        self.plot_widget.setPlotData(y_list,x_list,general_parameters_dict,other_parameters_dict)
        self.plot_widget.onClickedContinueDrawButton()

    #重新画图
    def onClickedRDraw(self):
        y_list, x_list = self.plot_data_widget.getData()
        general_parameters_dict, other_parameters_dict = self.plot_setting_widget.getParameters()
        self.plot_widget.setPlotData(y_list, x_list, general_parameters_dict, other_parameters_dict)
        self.plot_widget.onClickedReDrawButton()


    def closeEvent(self, QCloseEvent):
        '''
        使用terminate强制退出线程不安全，
        可能造成内存泄漏，
        但是目前还没有想到更好的方法。
        (守护线程thread有setDaemon方法，
        但是Qthread貌似没有。。)
        :param QCloseEvent:
        :return:
        '''
        reply = QMessageBox.question(self, "关闭确认", '是否要退出建模分析？', QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.timer_thread and self.timer_thread.isRunning():
                self.timer_thread.terminate()
                self.timer_thread.deleteLater()
            if self.worker_thread and self.worker_thread.isRunning():
                self.worker_thread.terminate()
                self.worker_thread.deleteLater()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelingWidget(1,2,3)
    window.show()
    sys.exit(app.exec_())
