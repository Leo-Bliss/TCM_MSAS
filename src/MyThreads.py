#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    :    2020/4/28 0028 16:04
# @Author  :    tb_youth
# @FileName:    MyThreads.py
# @SoftWare:    PyCharm
# @Blog    :    https://blog.csdn.net/tb_youth


from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtGui import QStandardItem
from PyQt5.QtCore import QThread, pyqtSignal
from openpyxl import workbook
import csv
from src import DataConverter, ExcelHelper

from algorthms import DSAPLS, LAPLS, RBMPLS, SEAPLS, PLSSDA, DBNPLS
from algorthms import MtreePLS, RFPLS, PLSCF, SBMPLS, GRAPLS


class ReaderExcelThread(QThread):
    standarModel_signal = pyqtSignal(QStandardItemModel)
    progressRate_signal = pyqtSignal(str)
    end_signal = pyqtSignal()

    def __init__(self, file_name, rows=25, columns=20):
        super(ReaderExcelThread, self).__init__()
        self.file_menu_name = file_name
        self.model = QStandardItemModel()
        self.min_rows, self.min_columns = rows, columns

    def run(self):
        # 这里读取数据返回列表便于表格中数据的更新
        self.progressRate_signal.emit("数据载入准备中...")
        excel_helper = ExcelHelper.ExcelHelper()
        data_list = excel_helper.read_excel(self.file_menu_name)
        if data_list == -1:
            return
        #print('start work!')
        cnt = len(data_list)
        for i, rows in enumerate(data_list):
            row = [QStandardItem(str(cell)) for cell in rows]
            self.model.appendRow(row)
            percent = int(i / cnt * 100 + 0.5)
            self.progressRate_signal.emit("数据载入进度:{}%".format(percent))

        # 自动填满，这样更加美观（不过增加了后期数据处理难度！）
        while self.model.rowCount() < self.min_rows:
            self.model.insertRows(self.model.rowCount(), 1)
        while self.model.columnCount() < self.min_columns:
            self.model.insertColumns(self.model.columnCount(), 1)

        # 数据加载完成
        self.progressRate_signal.emit("数据载入进度:100%")
        self.standarModel_signal.emit(self.model)
        #print('send finised')
        self.end_signal.emit()


class WriteExcelThread(QThread):
    start_signal = pyqtSignal(str)
    end_signal = pyqtSignal()

    def __init__(self, file_path, model):
        super(QThread, self).__init__()
        self.file_path = file_path
        self.model = model

    def run(self):
        self.start_signal.emit('导出准备中...')
        if isinstance(self.model, QStandardItemModel):
            data_converter = DataConverter.DataConverter()
            data_list = data_converter.model_to_list(self.model)
        elif isinstance(self.model, list):
            data_list = self.model
        else:
            return
        file_type = self.file_path.rsplit('.',maxsplit=1)[-1]
        if file_type == 'xlsx':
            self.write_xlsx(data_list)
        elif file_type == 'csv':
            self.write_csv(data_list)
        else:
            return


    def write_xlsx(self,data_list):
        wb = workbook.Workbook()
        wb.encoding = 'utf-8'
        wa = wb.active
        cnt = len(data_list)
        for i, item in enumerate(data_list):
            wa.append(item)
            self.start_signal.emit('导出进度:{}%'.format(int(i / cnt * 100)))
        wb.save(self.file_path)
        self.start_signal.emit('导出进度:100%')
        self.end_signal.emit()

    def write_csv(self,data_list):
        with open(self.file_path,'w',encoding='utf-8',newline='' "") as  file:
            writer = csv.writer(file)
            cnt = len(data_list)
            for i,row in enumerate(data_list):
                row_list = str(row).replace("'",'')[1:-1].split(',')
                writer.writerow(row_list)
                self.start_signal.emit('导出进度:{}%'.format(int(i / cnt * 100)))
            self.start_signal.emit('导出进度:100%')
            self.end_signal.emit()


class InitVarListThread(QThread):
    '''
        初始化设置变量列表线程
    '''
    init_var_list_signal = pyqtSignal(list)
    end_signal = pyqtSignal()

    def __init__(self, model):
        super(InitVarListThread, self).__init__()
        self.model = model

    def run(self):
        rows = self.model.rowCount()
        columns = self.model.columnCount()
        # 获取变量列表，self.model.index(row, column).data() not in ['',None]
        for row in range(rows):
            self.var_list = [self.model.index(row, column).data() for column in range(columns)
                             if self.model.index(row, column).data()]
            self.init_var_list_signal.emit(self.var_list)
            break
        self.end_signal.emit()



class TimerThread(QThread):
    '''
    运行计时线程
    '''
    start_signal = pyqtSignal(str)

    def __init__(self):
        super(TimerThread, self).__init__()
        self.sec = 0
        self.is_running = True

    def run(self):
        # 这里最好不要使用while True,要不然不好终止线程
        while self.is_running:
            self.start_signal.emit(str(self.sec))
            self.msleep(1000)
            self.sec += 1
        self.quit()


class WorkerThread(QThread):
    '''
    运行算法线程
    '''
    start_signal = pyqtSignal()
    end_signal = pyqtSignal()
    res_signal = pyqtSignal(dict)

    def __init__(self, model, all_dict, id):
        super(WorkerThread, self).__init__()
        self.model = model
        self.all_dict = all_dict
        self.id = id

    def run(self):
        # 运行算法和对应id构成字典:避免使用过多的if...else
        alg_dict = {
            0: DSAPLS.RunDSAPLS,
            1: LAPLS.RunLAPLS,
            2: RBMPLS.RunRBMPLS,
            3: SEAPLS.RunSEAPLS,
            4: PLSSDA.RunPLSSDA,
            5: DBNPLS.RunDBNPLS,
            6: MtreePLS.RunMtreePLS,
            7: RFPLS.RunRFPLS,
            8: PLSCF.RunPLSCF,
            9: SBMPLS.RunSBMPLS,
            10: GRAPLS.RunGRAPLS
        }

        self.start_signal.emit()
        data_converter = DataConverter.DataConverter()
        self.data_list = data_converter.model_to_list(self.model)
        if not self.data_list:
            return
        self.df = data_converter.list_to_DataFrame(self.data_list)
        alg = alg_dict[self.id](self.df, self.all_dict)
        alg.run()
        res_dict = alg.getRes()
        self.res_signal.emit(res_dict)
        self.end_signal.emit()


if __name__=='__main__':
    pass
