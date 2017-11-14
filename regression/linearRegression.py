# coding:utf-8
import happybase
from pyspark.ml.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.shell import sc
import numpy as np
import matplotlib.pyplot as plt
"""
    数据集不属于指数分布族,不能使用广义线性模型来进行回归分析
    R2总是大于1，尚无法解决
"""

def connect(ip,table_name):
    connection=happybase.Connection(ip)
    table=connection.table(table_name)
    return table


# 获取原始数据
def get_data(table):
    row_data=[]
    for k,v in  table.scan():
        row_data.append(v)
    return row_data


# 训练模型,返回预测值和实际值的rdd
def train_model(data):
    linear_model=LinearRegressionWithSGD.train(data,iterations=20,step=0.003,regType='l2',intercept=False)
    tp=data.map(lambda x:(x.label,linear_model.predict(x.features)))
    return tp


# 拟合优度
def get_R2(train,label_mean):
    linear_model= LinearRegressionWithSGD.train(train,iterations=30,step=0.009,intercept=False,regType='l2')
    SSR = train.map(lambda x: (linear_model.predict(x.features)-label_mean)**2).sum()
    SSE = train.map(lambda x: (x.label-linear_model.predict(x.features))**2).sum()
    SST=train.map(lambda x:(x.label-label_mean)**2).sum()
    tp=train.map(lambda x:(linear_model.predict(x.features),x.features))
    multiple_correlation=np.sqrt(SSR/SST)
    return (SSR/SST)


# 类别特征化为二元One-Hot  Encoding
def extract_features(record):
    mapping = [get_mapping(record, i) for i in range(1, 4)]
    cat_len = sum(map(len, mapping))
    cat_vec=np.zeros(cat_len)
    i=0
    step=0
    for field in record[1:4]:
        m=mapping[i]
        idx=m[field]
        cat_vec[idx+step]=1
        i=i+1
        step=step+len(m)
    return cat_vec


def get_mapping(rdd,idx):
    return rdd.map(lambda x:x[idx]).distinct().zipWithIndex().collectAsMap()


# 均方根误差
def squared_error(actual,pred):
    return (pred-actual)**2


# 绝对值误差
def abs_error(actual,pred):
    return np.abs(pred-actual)

# 标准化正态分布要求原来的数据也服从正态分布
def standardize(data):
    data=data.map(lambda x:map(float,[x[0],x[1],x[2],x[3]]))
    for t in range(1,4):
        mean = data.map(lambda x:x[t]).stats().mean()
        stdev = data.map(lambda x:x[t] - mean).stats().stdev()
        print(mean)
        print(stdev)
        print(t)
        if t==1:
            data=data.map(lambda x:[x[0],(x[1]-mean)/stdev,x[2],x[3]])
        if t==2:
            data=data.map(lambda x:[x[0],x[1],(x[2]-mean)/stdev,x[3]])
        if t==3:
            data=data.map(lambda x: [x[0], x[1], x[2],(x[3] - mean )/ stdev])
    return data

# 均方根对数误差
def squared_log_error(actual,pred):
    return (np.log(pred+1)-np.log(actual+1))**2


def data_hist(data,param=None):
    plt.hist(data)
    plt.xlabel('DO')
    plt.ylabel('amount')
    plt.title('DO_Hist')
    plt.show()

def evaluate(train,test,iterations,step,regParam,regType,intercept):
    pass

# 剔除重复数据
def unique(records):
    unique_key = list(set(records.keys()))
    unique_data = [(unique_key[i], records[unique_key[i]]) for i in range(len(unique_key))]
    data = sc.parallelize(unique_data)
    data = data.map(lambda x: LabeledPoint((x[0]), (x[1])))
    return data

# 相关系数
def correlation(data):
    COND_tem_covariance = np.cov(data.map(lambda x: x.label).collect(), data.map(lambda x: x.features[0]).collect())
    COND_ph_covariance = np.cov(data.map(lambda x: x.label).collect(), data.map(lambda x: x.features[1]).collect())
    COND_DO_covariance = np.cov(data.map(lambda x: x.label).collect(), data.map(lambda x: x.features[2]).collect())
    COND_tem_correlation = COND_tem_covariance[0][1] / (COND_tem_covariance[0][1] * COND_tem_covariance[1][1])
    COND_ph_correlation = COND_ph_covariance[0][1] / (COND_ph_covariance[0][1] * COND_ph_covariance[1][1])
    COND_DO_correlation = COND_DO_covariance[0][1] / (COND_DO_covariance[0][1] * COND_DO_covariance[1][1])

# R2总是大于1，解决不了
if  __name__=='__main__':
    table=connect('192.168.232.69','water_quality')
    row_data=sc.parallelize(get_data(table))
    re_data=row_data.map(lambda x:((x['info:COND']),(x['info:tem'],x['info:pH'],x['info:DO']))).collectAsMap()
    # 不用标准化,因为不服从正态分布
    # 标准化数据
    # std_data=standardize(re_data)
    # data=std_data.map(lambda x:LabeledPoint((x[0]),(x[1],x[2],x[3])))
    # label_mean=data.map(lambda x:x.label).stats().mean()
    # print(get_R2(data,label_mean))
    # 不用One-Hot  Encoding,因为都是数值类型
    # 非标准化的数据
    data=unique(re_data)
    label_mean = data.map(lambda x: x.label).stats().mean()

    print(np.sqrt(get_R2(data,label_mean)))
    # plt.subplot(2,1,1)
    # plt.plot(data.map(lambda x:x.label).collect())
    # tp=get_R2(data,label_mean)
    # plt.subplot(2,1,2)
    # plt.plot(tp.map(lambda x:x[0]).collect())
    # plt.show()

    # stats有count,mean,stdev,max,min,sum,va riance,samplexxx

    # train,test=std_data.randomSplit([0.8,0.2])
    # print(records.take(5))
    # print(np.sqrt(train_model(data).map(lambda (x,y):squared_error(x,y)).mean()))
    # print(np.sqrt(train_model(data).map(lambda (x,y):(squared_log_error(x,y))).mean()))

# {'info:salt': '', 'info:package_num': '850', 'info:tem': '18.41', 'info:COND': '2.06', 'info:node_num': '1', 'info:pH': '7.69', 'info:DO': '28.57', 'info:receive_time': '2017-03-06 10:28:27.0'}