#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time    :    2020/6/7 0007 9:28
#@Author  :    tb_youth
#@FileName:    fileHash.py
#@SoftWare:    PyCharm
#@Blog    :    https://blog.csdn.net/tb_youth

'''
测试对比文件的MD5来确定同一个文件
'''
import hashlib

def hashs(fineName, type="sha256", block_size=64 * 1024):
    '''
    Support md5(), sha1(), sha224(), sha256(), sha384(), sha512(), blake2b(), blake2s(),
    sha3_224, sha3_256, sha3_384, sha3_512, shake_128, and shake_256
    '''
    with open(fineName, 'rb') as file:
        hash = hashlib.new(type, b"")
        while True:
            data = file.read(block_size)
            if not data:
                break
            hash.update(data)
        return hash.hexdigest()



if __name__=='__main__':
    hash1 = hashs('./m1.py')
    hash2 = hashs('./space.png')
    print(hash1,hash2,hash1==hash2)