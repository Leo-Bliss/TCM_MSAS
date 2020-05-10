#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :    2020/3/8 0008 17:40
# @Author  :    tb_youth
# @FileName:    MainWindow.py
# @SoftWare:    PyCharm
# @Blog    :    https://blog.csdn.net/tb_youth

'''
MSAS: Mini Statistical Analysis System
主界面：
导入数据，
数据展示，
数据处理，
模型选择
'''

import sys, csv, io
from PyQt5.QtWidgets import QApplication, QWidget, QMenu, qApp, QShortcut
from PyQt5.QtWidgets import QTableView, QFileDialog, QStyleFactory
from PyQt5.QtWidgets import QMenuBar, QToolBar, QStatusBar, QAction
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QStandardItemModel, QPixmap, QIcon
from PyQt5.QtGui import QStandardItem, QColor, QCursor, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal, QObject

from src.SetVarsParametersWidget import SetParameterDialog
from src.FindWidget import FindWidget
from src.MyThreads import ReaderExcelThread, WriteExcelThread, InitVarListThread
from src.ModelingWidget import ModelingWidget


# 自定义的信号类，用于窗口通信
class MySignal(QObject):
    sender = pyqtSignal(QStandardItemModel)

    def send(self, model):
        self.sender.emit(model)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        '''
        res_pos:查找结果的位置
        focus_pos:鼠标聚焦的位置
        :param
        :return:
        '''
        # self.resize(1400,900)
        # self.setCenter()
        self.showMaximized()
        self.setWindowTitle('江中微型统计分析系统')
        self.res_pos = []
        self.focus_pos = None
        self.signal = MySignal()

        # 菜单栏
        self.menu_bar = QMenuBar()
        self.file_menu = self.menu_bar.addMenu('文件')
        self.edit_menu = self.menu_bar.addMenu('编辑')
        self.model_menu = self.menu_bar.addMenu('模型')
        self.view_menu = self.menu_bar.addMenu('视图')
        self.help_menu = self.menu_bar.addMenu('帮助')

        # 文件菜单下的子菜单
        self.open_action = self.file_menu.addAction('打开')
        self.save_action = self.file_menu.addAction('保存')
        self.file_menu.addSeparator()
        self.exit_action = self.file_menu.addAction('Exit')

        # 编辑菜单下的子菜单
        self.cut_action = self.edit_menu.addAction('剪切(X)')
        self.copy_action = self.edit_menu.addAction('复制(C)')
        self.paste_action = self.edit_menu.addAction('粘贴(V)')
        self.edit_menu.addSeparator()
        self.find_action = self.edit_menu.addAction('查找(F)')

        self.select_all_action = QAction('全选(A)')
        self.clear_action = QAction('清除(D)')
        self.clear_all_action = QAction('清空(Q)')

        # 模型菜单下的子菜单
        self.dsa_pls_action = self.model_menu.addAction('DSA-PLS')
        self.sbm_pls_action = self.model_menu.addAction('SBMPLS')
        self.model_menu.addSeparator()
        self.pls_cf_action = self.model_menu.addAction('PLSCF')
        self.la_pls_action = self.model_menu.addAction('LAPLS')
        self.gra_pls_action = self.model_menu.addAction('GRA-PLS')
        self.model_menu.addSeparator()
        self.rbm_pls_action = self.model_menu.addAction('RBM-PLS')
        self.sea_pls_action = self.model_menu.addAction('SEA-PLS')
        self.dbn_pls_action = self.model_menu.addAction('DBN-PLS')
        self.model_menu.addSeparator()
        self.mtree_pls_action = self.model_menu.addAction('Mtree-PLS')
        self.rf_pls_action = self.model_menu.addAction('RF-PLS')
        self.pls_s_da_action = self.model_menu.addAction('PLS-S-DA')

        # 快捷键
        self.open_action.setShortcut(QKeySequence(QKeySequence.Open))
        self.save_action.setShortcut(QKeySequence(QKeySequence.Save))
        self.find_action.setShortcut(QKeySequence(QKeySequence.Find))
        self.copy_action.setShortcut(QKeySequence(QKeySequence.Copy))
        self.cut_action.setShortcut(QKeySequence(QKeySequence.Cut))
        self.paste_action.setShortcut(QKeySequence(QKeySequence.Paste))
        self.select_all_action.setShortcut(QKeySequence(QKeySequence.SelectAll))
        # 这些不是标准的快捷键需要按下面的方式设置
        QShortcut(QKeySequence('Ctrl+D'), self).activated.connect(self.triggeredClear)
        QShortcut(QKeySequence('Ctrl+Q'), self).activated.connect(self.triggeredClearAll)

        # 视图菜单下的子菜单
        self.tool_view = QAction('工具栏', checkable=True)
        self.tool_view.setChecked(True)
        self.view_menu.addAction(self.tool_view)
        self.statu_view = QAction('状态栏', checkable=True)
        self.statu_view.setChecked(True)
        self.view_menu.addAction(self.statu_view)

        # 帮助菜单下的子菜单
        self.about_action = self.help_menu.addAction('关于')

        # 工具栏
        self.tool_bar = QToolBar()
        self.tool_bar.addAction(self.open_action)
        self.tool_bar.addAction(self.save_action)
        self.tool_bar.addAction(self.cut_action)
        self.tool_bar.addAction(self.copy_action)
        self.tool_bar.addAction(self.paste_action)
        self.tool_bar.addAction(self.find_action)

        # #tool文本显示在下方
        # self.tool_bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        # findWidget
        self.find_action_widget = FindWidget()
        self.find_action_widget.hide()

        # 表格
        # table_view
        self.init_rows, self.init_colums = 25, 20
        self.model = QStandardItemModel(self.init_rows, self.init_colums)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage('状态栏')

        # 创建布局
        layout = QVBoxLayout()
        layout.addWidget(self.menu_bar)
        layout.addWidget(self.tool_bar)
        layout.addWidget(self.find_action_widget)
        layout.addWidget(self.table_view)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)

        self.initRightMenu()
        self.connectSignal()
        self.prettifyUI()

    def initRightMenu(self):
        # 右键菜单栏
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.context_menu = QMenu()
        general_op_list = [self.copy_action, self.cut_action, self.paste_action]
        self.context_menu.addActions(general_op_list)
        self.context_menu.addSeparator()
        other_op_list = [self.select_all_action, self.clear_action, self.clear_all_action]
        self.context_menu.addActions(other_op_list)
        self.context_menu.addSeparator()
        self.addRow_down_action = self.context_menu.addAction('插入行(下)')
        self.addColumn_right_action = self.context_menu.addAction('插入列(右)')
        self.addRow_up_action = self.context_menu.addAction('插入行(上)')
        self.addColumn_left_action = self.context_menu.addAction('插入列(左)')
        self.context_menu.addSeparator()
        self.delRow_action = self.context_menu.addAction('删除行')
        self.delColumn_action = self.context_menu.addAction('删除列')
        self.context_menu.addSeparator()

    # 关联信号
    def connectSignal(self):
        # 文件
        self.open_action.triggered.connect(self.triggeredOpen)
        self.save_action.triggered.connect(self.triggeredSave)
        self.exit_action.triggered.connect(self.close)
        # 视图
        self.tool_view.triggered.connect(self.triggeredView)
        self.statu_view.triggered.connect(self.triggeredView)
        self.view_dic = {'工具栏': self.tool_bar, '状态栏': self.status_bar}
        # 查找（替换）框
        self.find_action.triggered.connect(self.triggeredFind)
        self.find_action_widget.search_action.triggered.connect(self.dataLocation)
        self.find_action_widget.down_aciton.triggered.connect(self.downAcitonLocation)
        self.find_action_widget.up_aciton.triggered.connect(self.upAcitonLocation)
        self.find_action_widget.close_aciton.triggered.connect(self.triggeredHideFind)
        self.find_action_widget.repalce_button.clicked.connect(self.onClickReplace)
        self.find_action_widget.repalceAll_button.clicked.connect(self.onClickReplaceAll)
        # 编辑
        self.copy_action.triggered.connect(self.triggeredCopy)
        self.paste_action.triggered.connect(self.triggeredPaste)
        self.clear_action.triggered.connect(self.triggeredClear)
        self.clear_all_action.triggered.connect(self.triggeredClearAll)
        self.cut_action.triggered.connect(self.triggeredCut)
        self.select_all_action.triggered.connect(self.table_view.selectAll)
        # 表格
        self.addRow_down_action.triggered.connect(lambda: self.addRow(1))
        self.addColumn_right_action.triggered.connect(lambda: self.addColumn(1))
        self.addRow_up_action.triggered.connect(lambda: self.addRow(0))
        self.addColumn_left_action.triggered.connect(lambda: self.addColumn(0))
        #一次行列删除暂时只支持删除一行或一列
        self.delRow_action.triggered.connect(lambda: self.model.removeRow(self.table_view.currentIndex().row()))
        self.delColumn_action.triggered.connect(
            lambda: self.model.removeColumn(self.table_view.currentIndex().column()))
        self.customContextMenuRequested.connect(self.rightMenuShow)
        # 算法模型
        self.dsa_pls_action.triggered.connect(lambda: self.commonTriggered(0, 'DSA-PLS'))
        self.la_pls_action.triggered.connect(lambda: self.commonTriggered(1, 'LAPLS'))
        self.rbm_pls_action.triggered.connect(lambda: self.commonTriggered(2, 'RBM-PLS'))
        self.sea_pls_action.triggered.connect(lambda: self.commonTriggered(3, 'SEA-PLS'))
        self.pls_s_da_action.triggered.connect(lambda: self.commonTriggered(4, 'PLS-S-DA'))
        self.dbn_pls_action.triggered.connect(lambda: self.commonTriggered(5, 'DBN-PLS'))
        self.mtree_pls_action.triggered.connect(lambda: self.commonTriggered(6, 'Mtree-PLS'))
        self.rf_pls_action.triggered.connect(lambda: self.commonTriggered(7, 'RF-PLS'))
        self.pls_cf_action.triggered.connect(lambda: self.commonTriggered(8, 'PLS-CF'))
        self.sbm_pls_action.triggered.connect(lambda: self.commonTriggered(9, 'SBMPLS'))
        self.gra_pls_action.triggered.connect(lambda: self.commonTriggered(10, 'GRA-PLS'))

    def initVarList(self, var_list):
        self.var_list = var_list

    # 获取参数对话框的相关参数:根据选择的算法，显示不同的设置界面
    def commonTriggered(self, id, name):
        self.init_var_list_thread = InitVarListThread(self.model)
        self.init_var_list_thread.init_var_list_signal.connect(self.initVarList)
        self.init_var_list_thread.end_signal.connect(lambda: self.initSetParametersUI(id, name))
        self.init_var_list_thread.start()

    def setParameter(self, id, dic):
        self.all_dict = dic
        print(self.all_dict)
        self.modeling_widget = ModelingWidget(self.model, self.all_dict, id,self)
        self.modeling_widget.show()


    # 设置算法的参数
    def initSetParametersUI(self, id, name):
        self.init_var_list_thread.quit()
        dialog = SetParameterDialog(id, self.var_list,self)
        dialog.setWindowTitle(name)
        dialog.sendSignal.sender.connect(self.setParameter)
        dialog.exec_()

    # 美化，icon
    def prettifyUI(self):
        icon = QIcon()
        icon.addPixmap(QPixmap('../imgs/打开.png'), QIcon.Normal, QIcon.Off)
        self.open_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/保存.png'), QIcon.Normal, QIcon.Off)
        self.save_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/剪切.png'), QIcon.Normal, QIcon.Off)
        self.cut_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/复制.png'), QIcon.Normal, QIcon.Off)
        self.copy_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/粘贴.png'), QIcon.Normal, QIcon.Off)
        self.paste_action.setIcon(icon)
        icon.addPixmap(QPixmap('../imgs/查找1.png'), QIcon.Normal, QIcon.Off)
        self.find_action.setIcon(icon)
        # 界面风格，logo
        QApplication.setStyle(QStyleFactory.keys()[2])
        self.setWindowIcon(QIcon('../imgs/school_logo.png'))

        # style_file = './style.qss'
        # qssStyle = CommonHelper.readQSS(style_file)
        # # print(qssStyle)
        # self.setStyleSheet(qssStyle)

    # 显示状态栏消息
    def showStatus(self, msg):
        self.status_bar.showMessage(msg)

    # 接收线程加载的数据
    def loadData(self, model):
        print('load...')
        self.model = model
        self.table_view.setModel(self.model)
        qApp.processEvents()

    def triggeredOpen(self):
        self.status_bar.showMessage('打开文件', 5000)
        file_name, _ = QFileDialog.getOpenFileName(self, '打开文件', '../data',
                                                   'AnyFile(*.*);;xlsx(*.xlsx);;csv(*.csv);;xls(*.xls)')
        if file_name:
            try:
                # 这里线程实例化一定要实例化成员变量，否则线程容易销毁
                self.thread = ReaderExcelThread(file_name)
                self.thread.standarModel_signal.connect(self.loadData)
                self.thread.progressRate_signal.connect(self.showStatus)
                self.thread.end_signal.connect(self.thread.quit)
                self.thread.start()
            except Exception as e:
                print(e)
                pass

    def triggeredSave(self):
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
            self.write_thread = WriteExcelThread(file_path, self.model)
            self.write_thread.start_signal.connect(self.showStatus)
            self.write_thread.end_signal.connect(self.write_thread.quit)
            self.write_thread.start()
            self.status_bar.showMessage('保存完毕！')
        except Exception as e:
            print(e)

    # 状态栏与工具栏的显示和隐藏
    def triggeredView(self, state):
        sender = self.sender().text()
        self.view_dic[sender].show() if state else self.view_dic[sender].hide()

    def triggeredFind(self):
        self.find_action_widget.show()

    # 重载信号，实现ESC隐藏查找窗口
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.find_action_widget.hide()

    # 聚焦到某个cell
    def positionFocus(self, x, y):
        self.table_view.verticalScrollBar().setSliderPosition(x)
        self.table_view.horizontalScrollBar().setSliderPosition(y)
        self.table_view.openPersistentEditor(self.model.index(x, y))
        self.table_view.setFocus()

    # 得到所以匹配项的位置
    def dataLocation(self):
        self.changeCellColor()
        text = self.find_action_widget.line_edit_find.text()
        self.res_pos = []
        flag = 0
        rows, columns = self.model.rowCount(), self.model.columnCount()
        try:
            for row in range(rows):
                for column in range(columns):
                    if text == self.model.index(row, column).data():
                        self.res_pos.append((row, column))
                        item = self.model.item(row, column)
                        item.setBackground(QColor(255, 255, 0))
                        item.setForeground(QColor(255, 0, 0))
                        # 转到到第一个匹配值的位置，并处于可编辑状态
                        if not flag:
                            flag = 1
                            self.positionFocus(row, column)
                            self.focus_pos = 0
        except Exception as e:
            print(e)

    # 向下跳转
    def downAcitonLocation(self):
        cnt = len(self.res_pos)
        if cnt == 0 or self.focus_pos == cnt - 1:
            return
        try:
            self.table_view.closePersistentEditor(
                self.model.index(self.res_pos[self.focus_pos][0], self.res_pos[self.focus_pos][1]))
            x, y = self.res_pos[self.focus_pos + 1]
            self.positionFocus(x, y)
            self.focus_pos += 1
        except Exception as e:
            print(e)

    # 向上跳转
    def upAcitonLocation(self):
        cnt = len(self.res_pos)
        if cnt == 0 or self.focus_pos == 0:
            return
        try:
            self.table_view.closePersistentEditor(
                self.model.index(self.res_pos[self.focus_pos][0], self.res_pos[self.focus_pos][1]))
            x, y = self.res_pos[self.focus_pos - 1]
            self.positionFocus(x, y)
            self.focus_pos -= 1
        except Exception as e:
            print(e)

    # 查找框隐藏
    def triggeredHideFind(self):
        self.changeCellColor()
        self.find_action_widget.hide()

    # 恢复cell的原色
    def changeCellColor(self):
        if self.res_pos is not None and len(self.res_pos):
            self.table_view.closePersistentEditor(
                self.model.index(self.res_pos[self.focus_pos][0], self.res_pos[self.focus_pos][1]))
            for item in self.res_pos:
                x, y = item
                item = self.model.item(x, y)
                item.setBackground(QColor(255, 255, 255))
                item.setForeground(QColor(0, 0, 0))

    # 单个匹配cell替换
    def onClickReplace(self):
        cnt = len(self.res_pos)
        text = self.find_action_widget.line_edit_replace.text()
        if self.res_pos is None or cnt == 0:
            return
        try:
            x, y = self.res_pos[self.focus_pos]
            self.model.setItem(x, y, QStandardItem(text))
        except Exception as e:
            print(e)

    # 全部匹配cell替换
    def onClickReplaceAll(self):
        cnt = len(self.res_pos)
        if self.res_pos is None or cnt == 0:
            return
        try:
            text = self.find_action_widget.line_edit_replace.text()
            for x, y in self.res_pos:
                self.model.setItem(x, y, QStandardItem(text))
        except Exception as e:
            print(e)

    # 1:当前行的下方添加一行,1:当前行的上方添加一行
    def addRow(self, i):
        self.model.insertRows(self.table_view.currentIndex().row() + i, 1)

    # 1：当前列的右方添加一行，0：当前列的左方添加一行
    def addColumn(self, i):
        self.model.insertColumns(self.table_view.currentIndex().column() + i, 1)

    # 显示右键菜单
    def rightMenuShow(self):
        # QCursor.pos() ：菜单显示的位置
        self.context_menu.popup(QCursor.pos())
        self.context_menu.show()

    # 复制
    def triggeredCopy(self):
        selection = self.table_view.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            QApplication.clipboard().setText(stream.getvalue())

    # 粘贴
    def triggeredPaste(self):
        selection = self.table_view.selectedIndexes()
        if selection:
            start_index = selection[0]
            # print(start_index.row(),start_index.column())
            buffer = QApplication.clipboard().text()
            reader = csv.reader(io.StringIO(buffer), delimiter='\t')
            max_row = self.model.rowCount() - 1
            max_column = self.model.columnCount() - 1
            for i, row in enumerate(reader):
                line = row[0]
                line_list = line.split(',')
                for j, cell in enumerate(line_list):
                    x = start_index.row() + i
                    y = start_index.column() + j
                    if x > max_row or y > max_column:
                        continue
                    self.model.setData(self.model.index(x, y), cell)

    # 清除
    def triggeredClear(self):
        select_indexs = self.table_view.selectedIndexes()
        for cell in select_indexs:
            self.model.setData(self.model.index(cell.row(), cell.column()), None)

    # 剪切
    def triggeredCut(self):
        self.triggeredCopy()
        self.triggeredClear()

    # 清空
    def triggeredClearAll(self):
        self.model = QStandardItemModel(self.init_rows, self.init_colums)
        self.table_view.setModel(self.model)

    # 窗口移动到正中央
    def setCenter(self):
        screen = QApplication.desktop()
        size = self.geometry()
        new_left = (screen.width() - size.width()) / 2
        new_top = (screen.height() - size.height()) / 2
        self.move(new_left, new_top)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
