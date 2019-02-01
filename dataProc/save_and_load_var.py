# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:33:20 2018

@author: lanliying

save and load var methods
"""


import numpy as np
a = np.random.randint(0, 100, size=(10, 5))
print(a.dtype, a.shape)  # int32 (10000, 5000) 下同

##############################################
#a.bin	190MB	1.130s	0.110s	需要处理type和reshape
a.tofile('a.bin')       #按照行连续存储为一维数组
b = np.fromfile('a.bin', dtype=np.int32)  # 需要设置正确dtype
print(b.shape)           # (50000000,) 读入数据是一维的，需要reshape
b_reshape = np.reshape(b,(10,5))        #按照行连续重新排列
print(a == b_reshape)   #逐个比较两个变量的数值是否相等
print((a == b_reshape).all()) 
print(a is b_reshape)   #比较内存地址是否一致，从而判定两者是否是同一个变量；
#但是，只要是在-5~256之间的整形，python不会给变量初始化新的内存空间，但是一旦超出256，则会分配新的空间。

##############################################
#.npy格式，np.save() 和 np.load() 
#numpy专用的二进制格式保存数据，能自动处理变量type和size
#a.npy	190MB	1.105s	0.124s
np.save('a.npy', a)#按照数组原始维度存储
c = np.load('a.npy')#按照数组原始维度读取
c_row,c_col = c.shape
print(c.shape)           # (10000, 5000)
print(c_row,c_col)           # (10000, 5000)


##############################################
#.txt格式 (或.csv/.xlsx)，np.savetxt() 和 np.loadtxt() 
#csv或xlsx可以借助python其他库工具实现。csv只能存数字，xlsx可以存数字字符
#a.txt	138MB	10.507s	60.225s	需要处理type
np.savetxt('a.txt', a, fmt='%d', delimiter=',') #设置以整数存储，以逗号隔开
d = np.loadtxt('a.txt', delimiter=',')
print(b.shape)           # (10000, 5000)



##############################################
#数组->字典(matlab可通用！)
from sklearn.datasets import make_classification
x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)
x_data_train = x[:800,:]
x_data_test = x[800:,:]
y_data_train = y[:800]
y_data_test = y[800:]

from scipy import io
io.savemat('x_data_train.mat',{'matrix':x_data_train})
io.savemat('x_data_test.mat',{'matrix':x_data_test})
io.savemat('y_data_train.mat',{'matrix':y_data_train})
io.savemat('y_data_test.mat',{'matrix':y_data_test})

#字典->数组
temp = io.loadmat('x_data_train.mat')
x_data_train = temp['matrix']

temp = io.loadmat('x_data_test.mat')
x_data_test = temp['matrix']

temp = io.loadmat('y_data_train.mat')
y_data_train = temp['matrix']

temp = io.loadmat('y_data_test.mat')
y_data_test = temp['matrix']

x = np.vstack((x_data_train,x_data_test))
y = np.hstack((y_data_train,y_data_test))
positive_x1 = [x[i,0] for i in range(1000) if y[0,i] == 1]
positive_x2 = [x[i,1] for i in range(1000) if y[0,i] == 1]
negetive_x1 = [x[i,0] for i in range(1000) if y[0,i] == 0]
negetive_x2 = [x[i,1] for i in range(1000) if y[0,i] == 0]



