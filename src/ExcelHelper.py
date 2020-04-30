#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/3/8 0008 17:53
#@Author  :    tb_youth
#@FileName:    ExcelHelper.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
表格的读写
'''
import csv
import xlrd
from  openpyxl import workbook

class ExcelHelper:
    def __init__(self):
        pass

    def read_excel(self,path):
        name, type = path.rsplit('.', maxsplit=1)
        if type == 'csv':
            with open(path, 'r',newline='',encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                data = [line for line in reader]
                return data
        elif type in ['xlsx', 'xls']:
            workbook = xlrd.open_workbook(path)
            sheet1 = workbook.sheet_by_index(0)
            rows = len(sheet1.col_values(0))
            data = [sheet1.row_values(i) for i in range(rows)]
            return data
        else:
            print('{}文件格式不支持'.format(type))
            return -1


    def write_excel(self,file_path, data_list):
        try:
            wb = workbook.Workbook()
            wb.encoding = 'utf-8'
            wa = wb.active
            for item in data_list:
                wa.append(item)
            wb.save(file_path)
        except Exception as e:
            print(e)

if __name__=='__main__':
    helper = ExcelHelper()
    path = r'../data/data2.xlsx'
    data = helper.read_excel(path)
    print(data)