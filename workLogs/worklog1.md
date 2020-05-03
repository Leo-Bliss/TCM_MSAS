问题
==
## 报错：
TypeError: ufunc 'isfinite' not supported for the input types, and the input

## 原因：
需要的数据类型不对，一般就是numpy里面的数据结构，尝试修改这个部分：
  x0 = mat(x0)
  y0 = mat(y0)
## 解决：
修改为：
  x0 = mat(x0,dtype=np.float64)
  y0 = mat(y0,dtype=np.float64)
  
 
## 报错
TypeError: 1oop of ufunc does not support argument 0 of type float which
## 解决
解决方案也和上面类似，都是找到需要用的数据，数据类型设置为np.float64

## mat 和 matrix 的区别
+ b = np.mat(a): 当a本身是matrix类型时，则b是a的引用
+ b = matrix(a): 不管a本身是什么类型，产生一个新的matrix,b指向它

## python作用域
Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域。
其它的代码块（如 if/elif/else/、try/except、for/while等）不会引入新的作用域，
也因此这些语句内定义的变量，外部也可以直接访问。
