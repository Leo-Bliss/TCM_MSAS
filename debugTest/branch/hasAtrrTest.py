#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/5/3 0003 12:47
#@Author  :    tb_youth
#@FileName:    hasAtrrTest.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
判断类的属性和方法是否存在

'''

class A:
    value = 1
    def __init__(self):
        pass
    def getValue(self):
        print('value:{}'.format(self.value))

    def setValue(self,value):
        self.value = value


if __name__=='__main__':
    a = A
    # way1:hasattr
    op = hasattr(a,'getValue')
    print(op)
    print(hasattr(a,'AA'))
    print(hasattr(a,'value'))
    
    # way2:dir
    # 属性，方法列表
    print(dir(a))
    print('value' in dir(a))