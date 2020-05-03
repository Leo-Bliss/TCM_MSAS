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
from PyQt5.QtWidgets import QApplication, QTabWidget, QStyleFactory
from PyQt5.QtWidgets import QWidget, QHBoxLayout,QMessageBox
from PyQt5.QtGui import QIcon

from src.PlotDataWidget import PlotDataWidget
from src.PlotSettingWidget import PlotSettingWidget
from src.PlotWidget import PlotWidget
from src.BriefWidget import BriefWidget
from src.MyThreads import TimerThread,WorkerThread

class ModelingWidget(QWidget):
    def __init__(self,model,all_dict,id):
        super(ModelingWidget,self).__init__()
        self.model = model
        self.all_dict = all_dict
        self.id = id
        self.initUI()

    def initUI(self):
        self.resize(1400,800)
        QApplication.setStyle(QStyleFactory.keys()[2])
        self.setWindowIcon(QIcon('../imgs/school_logo.png'))
        self.setWindowTitle('建模分析')
        self.timer_thread = None
        self.worker_thread = None
        hlayout = QHBoxLayout()
        self.tab_widget = QTabWidget()

        self.brief_widget = BriefWidget()
        self.tab_widget.addTab(self.brief_widget, '摘要')

        self.plot_data_widget = PlotDataWidget()
        self.tab_widget.addTab(self.plot_data_widget, '数据')

        self.plot_setting_widget = PlotSettingWidget()
        self.tab_widget.addTab(self.plot_setting_widget, '图形')

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
        try:
            self.work()
        except:
            pass

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
        self.brief_widget.progressbar.show()
        self.brief_widget.status_bar.show()
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
        var_dict = self.all_dict.get('var_dict')
        independ_var_list = var_dict.get('independ_var')
        depend_var_list = var_dict.get('depend_var')
        cnt1 = len(independ_var_list)
        cnt2 = len(depend_var_list)
        show_independ = independ_var_list if cnt1 <= 10 else independ_var_list[0:10]
        show_depend = depend_var_list if cnt2 <= 10 else depend_var_list[0:10]
        tmp_x = str(show_independ)[1:-1]
        #为了美观，暂时最多显示10个变量
        if cnt1 > 10:
            tmp_x += '...'
        tmp_y = str(show_depend)[1:-1]
        if cnt2 > 10:
            tmp_y += '...'
        var_info = '自变量：{}\n因变量：{}'.format(tmp_x, tmp_y)
        self.brief_widget.appendText(var_info)

    def showMainResInfo(self,dct):
        self.brief_widget.appendText('-'*55)
        for k,v in dct.items():
            line = '{}：{}'.format(k,v)
            self.brief_widget.appendText(line)
        self.brief_widget.appendText('-'*55)


    def endRun(self):
        # 不会更好的终止方案。。。
        if self.worker_thread:
            self.worker_thread.terminate()
        if self.timer_thread:
            self.timer_thread.is_running = False
        self.brief_widget.setStatus('已经终止建模...')
        self.brief_widget.progressbar.close()
        self.brief_widget.termination_btn.setEnabled(False)

    def hideProgressBar(self):
        self.brief_widget.setStatus('建模已完成...')
        self.brief_widget.termination_btn.setEnabled(False)
        self.showVarInfo()
        self.brief_widget.appendText('-'*55)
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
        timer = None #self.timer_thread
        worker = self.worker_thread
        reply = QMessageBox.question(self, "关闭确认", '是否要退出建模分析？', QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if timer and timer.isRunning():
                timer.terminate()
                timer.deleteLater()
            if worker and worker.isRunning():
                worker.terminate()
                worker.deleteLater()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelingWidget(1,2,3)
    window.show()
    sys.exit(app.exec_())
