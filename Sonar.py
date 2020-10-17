#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

path = 'xxxxxxx'#将xxxxxxx替换为数据地址
df = pd.read_csv(path,header = None)

#第一类数据集 共97个样本 60维（不含标签）
sonar1 = df.values[0:97,0:60]
#第二类数据集 共111个样本 60维（不含标签）
sonar2 = df.values[97:208,0:60]

#数据预处理
#数据按类别打乱
shuffle = np.random.permutation(sonar1.shape[0])#打乱后的行号
new_sonar1 = sonar1[shuffle]

shuffle = np.random.permutation(sonar2.shape[0])#打乱后的行号
new_sonar2 = sonar2[shuffle]

#数据拆分为训练集和测试集,按照3:2配置（多余的放入测试集中）
    #sonar1 训练 58个 测试 39个
train_sonar1 = new_sonar1[0:58]
test_sonar1 = new_sonar1[58:97]

    #sonar2 训练 66个 测试 45个
train_sonar2 = new_sonar2[0:66]
test_sonar2 = new_sonar2[66:111]

#均值向量
mean1 = np.mean(train_sonar1,axis = 0)
mean2 = np.mean(train_sonar2,axis = 0)

#计算类内离散度矩阵
sw1 = np.zeros((60,60))#0矩阵，60*60
for x1 in train_sonar1:
    a = x1-mean1
    a = a.reshape(-1,1)#修正为列向量60*1
    sw1 = sw1+np.dot(a,a.T)
print("类内离散度矩阵Sw1\n",sw1,"\n")

sw2 = np.zeros((60,60))
for x2 in train_sonar2:
    a = x2-mean2
    a = a.reshape(-1,1)#修正为列向量60*1
    sw2 = sw2+np.dot(a,a.T)
print("类内离散度矩阵Sw2\n",sw2,"\n")

#总类内离散度矩阵
sw = sw1+sw2
print("总类内离散度矩阵Sw\n",sw,"\n")
sw = np.array(sw,dtype = 'float')

#计算投影向量
x = np.array([mean1-mean2])
x = x.T
w = np.dot(np.linalg.inv(sw),x)#列向量
print("投影向量w\n",w,"\n")


#计算LDA二分类线性判别时的阈值y0
#降维空间下的不同类的均值
m1 = np.dot(w.T,mean1)
m2 = np.dot(w.T,mean2)
y0 = (m1+m2)/2#阈值
print("阈值\n",y0,"\n")

#进行测试
class1 = 0#正确数初始化为0
class2 = 0

for x1 in test_sonar1:
    y = np.dot(w.T,x1)
    if y>=y0:
        class1+=1

for x2 in test_sonar2:
    y = np.dot(w.T,x2)
    if y<y0:
        class2+=1

classification_accuracy = (class1+class2)/(45+39)
print("测试集上的准确率:",classification_accuracy*100,'%')


# In[ ]:




