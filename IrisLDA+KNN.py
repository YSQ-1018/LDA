#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd #读取数据
from sklearn.preprocessing import LabelEncoder #将class(三种花的名字)编码为数值
import numpy as np

#数据读入与预处理
iris_data_path = 'xxxxxx'#将xxxxx更换为存放数据的地址

df = pd.read_csv(iris_data_path,names = ['sepal length','sepal width','petal length','petal width','class'])

print(df.tail())
print(df.head())

#为类别编码
encoder = LabelEncoder().fit(df['class'])
df['class'] = encoder.transform(df['class'])
print(df.head())

#数据按类分配
iris1 = df.values[0:50,0:4]
iris2 = df.values[50:100,0:4]
iris3 = df.values[100:150,0:4]

#数据按类别打乱
def shuffle_data(iris):
    """
        iris----传入的数据集
        返回打乱的数据集
    """
    per = np.random.permutation(iris.shape[0])#打乱后的行号
    new_iris = iris[per,:]
    return new_iris

new_iris1 = shuffle_data(iris1) #获取打乱后的数据

new_iris2 = shuffle_data(iris2)

new_iris3 = shuffle_data(iris3)


#数据分为训练集和测试集，按照4:1配置，即每次随机在每一类中取40个样本作为训练集，剩余的10个样本作为测试集
train_iris1 = new_iris1[0:40]
train_iris2 = new_iris2[0:40]
train_iris3 = new_iris3[0:40]


test_iris1 = new_iris1[40:50]#测试集
test_iris2 = new_iris2[40:50]
test_iris3 = new_iris3[40:50]


# ## 上面已经进行了数据的打乱与训练集和测试集的划分

# ## 下面开始LDA算法

# ### 1、计算均值向量

# In[12]:


#均值向量
def mean_vector(train_iris):
    """
        train_iris----传入的数据集
        返回该数据集的均值向量
    """
    mean = np.mean(train_iris,axis = 0)
    return mean

#原来只用了训练集中的样本计算均值，现在使用全体数据计算均值
mean1 = mean_vector(new_iris1)
mean2 = mean_vector(new_iris2)
mean3 = mean_vector(new_iris3)

print("mean vector1:",mean1)
print("mean vector2:",mean2)
print("mean vector3:",mean3)


# ### 2、计算类内离散度矩阵

# In[13]:


#原来此处我也只计算了训练集数据的Sw，现在传入了整体数据
#计算类内离散度矩阵
def with_in_scatter_matrix(mean1, train_iris1):
    """
        mean1----传入数据集的均值向量
        train_iris1----传入的数据集名称
        返回该数据集（同一类）的类内离散度矩阵
    """
    Sw1 = np.zeros((4,4))#初始设为空矩阵，shape是4*4
    for x1 in train_iris1:
        a = x1-mean1
        a = a.reshape(-1,1)
        b = a.T
        #计算第一类的类内离散度矩阵
        Sw1 = Sw1 + np.dot(a,b)
    return Sw1

Sw1 = with_in_scatter_matrix(mean1, new_iris1)

Sw2 = with_in_scatter_matrix(mean2, new_iris2)

Sw3 = with_in_scatter_matrix(mean3, new_iris3)

#总类内离散度矩阵
Sw = Sw1 + Sw2 +Sw3

print("第一类类内离散度矩阵\n",Sw1,"\n")
print("第二类类内离散度矩阵\n",Sw2,"\n")
print("第三类类内离散度矩阵\n",Sw3,"\n")
print("总类内离散度矩阵\n",Sw,"\n")


# ## 3、计算类间离散度矩阵

# In[14]:


#类间离散度矩阵

     #训练集样本个数
N1 = new_iris1.shape[0]
N2 = new_iris2.shape[0]
N3 = new_iris3.shape[0]

    #定义m为不考虑标签的整体训练样本的均指向量
x = np.zeros((1,4))
for x1 in new_iris1:
    x += x1
for x2 in new_iris2:
    x += x2
for x3 in new_iris3:
    x += x3

m = x/(N1+N2+N3)#m为整体训练集的均值向量

    #定义类间离散度矩阵为Sb
def between_scatter_matrix(N1, m, mean1):
    """
        N1----数据集的样本个数
        m----全体训练集的均值向量
        mean1----某一类训练集的均指向量
        返回类间离散度矩阵的一个分量
    """
    Sb1 = N1*np.dot((mean1-m).T,(mean1-m))
    return Sb1

Sb1 = between_scatter_matrix(N1, m, mean1)
Sb2 = between_scatter_matrix(N2, m, mean2)
Sb3 = between_scatter_matrix(N3, m, mean3)

Sb = Sb1+Sb2+Sb3#类间离散度矩阵

print("类间离散度矩阵\n",Sb,"\n")


# ## 4、求解广义特征值问题

# In[15]:


#求解广义特征值问题
eigenvalues,eigenvectors = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

#由于需要的是列向量，因此先对该矩阵取转置，再取行向量，最终可得到其列向量
eigenvectors = eigenvectors.T

print("特征值：",eigenvalues,"\n")
print("特征向量:",eigenvectors,"\n")

dic = {eigenvalues[0]:eigenvectors[0],
       eigenvalues[1]:eigenvectors[1],
       eigenvalues[2]:eigenvectors[2],
       eigenvalues[3]:eigenvectors[3]}

#为字典按key值排序
dic = sorted(dic.items(),key = lambda asd:asd[0],reverse=True)
for eigen in dic:
    print(eigen)


# ## 5、求解投影向量（矩阵）

# In[16]:


#三类问题取前两个特征向量作为投影矩阵

x1 = np.array(dic[0][1].reshape(1,-1),dtype = 'float')#注意此处应声明该向量的类型，否则下面的M处的操作会报错
x2 = np.array(dic[1][1].reshape(1,-1),dtype = 'float')
W = np.array([x1,x2])
W = W.reshape(W.shape[0],-1)#矩阵维度修改
print("投影向量为\n",W[0],"\n",W[1])
print(W.shape)


# ## 6、对训练集和测试集进行LDA降维

# In[17]:


def LDA(dataset,W):
    """
        对某一数据集执行LDA降维
    """
    tmp = np.dot(W,dataset.T)
    return tmp.T

train_iris1_after_LDA = LDA(train_iris1,W)
train_iris2_after_LDA = LDA(train_iris2,W)
train_iris3_after_LDA = LDA(train_iris3,W)

test_iris1_after_LDA = LDA(test_iris1,W)
test_iris2_after_LDA = LDA(test_iris2,W)
test_iris3_after_LDA = LDA(test_iris3,W)

print("训练集和测试集LDA降维完成!")
print("训练集shape:",train_iris1_after_LDA.shape)
print("测试集shape:",test_iris1_after_LDA.shape)

print(test_iris1_after_LDA)
#print(test_iris2_after_LDA)
#print(test_iris3_after_LDA)


new_iris1_LDA = LDA(new_iris1,W)
new_iris2_LDA = LDA(new_iris2,W)
new_iris3_LDA = LDA(new_iris3,W)

#绘图
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#取第一列
x1 = new_iris1_LDA[:,0]

#取第二列
x2 = new_iris1_LDA[:,1]

y1 = new_iris2_LDA[:,0]

y2 = new_iris2_LDA[:,1]

z1 = new_iris3_LDA[:,0]

z2 = new_iris3_LDA[:,1]


plt.scatter(x1,x2,marker = 'x',color = 'red',s = 40,label = 'Iris-setosa')

plt.scatter(y1,y2,marker = '+',color = 'blue',s = 40,label = 'Iris-Versicolor')

plt.scatter(z1,z2,marker = 'o',color = 'green',s = 40,label = 'Iris-Virginica')


plt.xlabel('LD1')
plt.ylabel('LD2')

leg = plt.legend(loc='upper left', fancybox=True)


# ## 7、KNN算法对数据进行分类

# In[18]:


#计算欧氏距离
def Euclidean_Dsitance(x,train):
    """
        x----测试集中的一个样本
        train----训练集
        返回一个列表，包含x到train所有样本的欧氏距离
    """
    ED = []
    for y in train:
        tmp = x - y
        d = np.linalg.norm(tmp,ord=2)
        ED.append(d)
    ED.sort()
    return(ED)
   


# ## 8、KNN算法

# In[19]:


def generate_label(k,m):
    """
        生成k个含相同元素的列表
        m为1,2,3其中一个
    """
    label = []
    for i in range(k):
        label.append(m)
    return label


def vote(vote_list):
    """
        执行投票过程
    """
    #计数器初始化
    c1,c2,c3 = 0,0,0
    for i in range(len(vote_list)):
        if vote_list[i]==1:
            c1+=1
        if vote_list[i]==2:
            c2+=1
        if vote_list[i]==3:
            c3+=1
    return c1,c2,c3
        
def KNN(k,test_set1,train_set1=train_iris1_after_LDA,train_set2=train_iris2_after_LDA,train_set3=train_iris3_after_LDA):
    """
        k为超参数，表示挑选的临近元素个数
        test_set为传入的LDA降维过的测试集
        返回整体投票列表
    """
    #全体样本的投票记录
    total_vote = []
    for sample in test_set1:
        #逐个计算测试样本与训练集样本之间的欧氏距离
        ed1 = Euclidean_Dsitance(sample[0],train_set1)
        min_distence_from_1 = ed1[:k]
        ed2 = Euclidean_Dsitance(sample[0],train_set2)
        min_distence_from_2 = ed2[:k]
        ed3 = Euclidean_Dsitance(sample[0],train_set3)
        min_distence_from_3 = ed3[:k]   
        #生成标签
        label1 = generate_label(k,1)
        label2 = generate_label(k,2)
        label3 = generate_label(k,3)
        dic = {}
        for i in range (len(label1)):
            dic[min_distence_from_1[i]]=label1[i]
            dic[min_distence_from_2[i]]=label2[i]
            dic[min_distence_from_3[i]]=label3[i]
        dic = sorted(dic.items(),key = lambda asd:asd[0])
        #print(dic)
        #生成投票列表
        vote_list = []
        for i in range(k):
            vote_list.append(dic[i][1])
        #print(vote_list)
        c1,c2,c3 = vote(vote_list)
        x = [c1,c2,c3]
        total_vote.append(x)
    print(total_vote)
    return total_vote

#分类正确的样本
correct = 0
correct1,correct2,correct3 = 0,0,0

vote1 = KNN(3,test_iris1_after_LDA)
for i in range(len(vote1)):
    if vote1[i][0]==3 or vote1[i][0]==2:
        correct+=1
        correct1+=1

vote2 = KNN(3,test_iris2_after_LDA)
for i in range(len(vote2)):
    if vote2[i][1]==3 or vote2[i][1]==2:
        correct+=1
        correct2+=1

vote3 = KNN(3,test_iris3_after_LDA)
for i in range(len(vote3)):
    if vote3[i][2]==3 or vote3[i][2]==2:
        correct+=1
        correct3+=1


# ## 9、打印结果

# In[20]:


print("第一类样本个数为:",len(vote1)) 
print("分类器将第一类样本分类正确的个数:",correct1)
print("第一类正确率:",correct1/len(vote1)*100,"%")
print("第二类样本个数为:",len(vote2)) 
print("分类器将第二类样本分类正确的个数:",correct2)
print("第一类正确率:",correct2/len(vote2)*100,"%")
print("第三类样本个数为:",len(vote3)) 
print("分类器将第三类样本分类正确的个数:",correct3)
print("第三类正确率:",correct3/len(vote3)*100,"%")
print("Average Accuracy:",(correct1/len(vote1)*100+
      correct2/len(vote2)*100+
      correct3/len(vote3)*100)/3,"%")


print("分类器将测试集全体样本分类正确的个数为",correct)
total = len(vote1)+len(vote2)+len(vote3)
print("测试集样本个数:",total)
print("Overall Accuracy:",correct/total*100,"%")


# In[ ]:




