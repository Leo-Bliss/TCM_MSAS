#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/8 0008 17:43
#@Author  :    tb_youth
#@FileName:    DataConverter.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
辅助数据处理：
数据类型的转换，
主要是QStandardItemModel，list，DataFrame之间的转换
'''

import pandas as pd

class DataConverter():
    '''
    说明：
    目前list_to_DataFrame仅支持
    原本list的二维形态就不是DataFrame的样式
    '''
    def __init__(self):
        pass

    def list_to_DataFrame(self,lst):
        df = pd.DataFrame(lst)
        column_list = list(df.iloc[0, 0:])
        data = df.iloc[1:, 0:]
        data.columns = column_list
        #index 下标从0开始
        data.index = pd.Series(range(len(lst)-1))
        return data

    def DataFrame_to_list(self,df):
        data_list = [[''] + df.columns.values.tolist()]
        index_list = df.index.values.tolist()
        for i, item in enumerate(df.values.tolist()):
            data_list.append([index_list[i]] + item)
        return data_list

    def DataFrame_to_list2(self, df):
        data_list = [df.columns.values.tolist()]
        for i, item in enumerate(df.values.tolist()):
            data_list.append(item)
        return data_list

    def judege_num(self,num):
        try:
            float(num)
            return True
        except:
            return False

    def model_to_list(self, model):
        if not hasattr(model,'rowCount'):
            return None
        rows = model.rowCount()
        columns = model.columnCount()
        data_list = []
        for row in range(rows):
            row_values = []
            for column in range(columns):
                cell = model.index(row, column).data()
                if cell is None:
                    break
                if self.judege_num(cell):
                    # or : cell.find('.') != -1
                    cell = float(cell) if '.'in cell else int(cell)
                row_values.append(cell)
            if len(row_values) == 0:
                continue
            data_list.append(row_values)
        return data_list

if __name__=='__main__':
   pass


