#coding:utf-8
import happybase
from pyspark import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.shell import  sc
import numpy as np
import matplotlib.pyplot as plt

def connect(ip,table_name):
    connection=happybase.Connection(ip)
    table=connection.table(table_name)
    return table


def get_data(table):
    row_data=[]
    for key,value in table.scan():
        row_data.append(value)
    return row_data


def extract_features(records,positions):
    features=map(float,[records[positions[0]],records[positions[1]],records[positions[2]]])
    return  features


def extract_label(records,position):
    return float(records[position])

# 训练模型
def train_model(train,maxDepth,maxBins):
    model=DecisionTree.trainRegressor(train,{},impurity='variance',maxDepth=maxDepth,maxBins=maxBins)
    return model

# 预测
def predict(model,test):
    pred=model.predict(test.map(lambda x:x.features))
    return pred

# True-vs-Prediction
def get_tp(model,data):
    pred = model.predict(data.map(lambda x:x.features))
    actual=data.map(lambda x:x.label)
    tp=actual.zip(pred)
    return tp

# 拟合优度
def get_R2(model,data):
    tp=get_tp(model,data)
    mean=data.map(lambda x:x.label).mean()
    SSR = tp.map(lambda (x,y): (y - mean) ** 2).sum()
    SST = tp.map(lambda (x,y) : (x - mean) ** 2).sum()
    SSE=tp.map(lambda (x,y):(x-y)**2).sum()
    return SSR / SST

def squared_error(pred,actual):
    return (pred-actual)**2


def abs_error(pred,actual):
    return np.abs(pred-actual)


def squared_log_error(pred,actual):
    return (np.log(pred+1)-np.log(actual+1))**2
# 这个+1不知道为什么


def evaluate(train,test,maxDepth,maxBins):
    model=DecisionTree.trainRegressor(train,{},impurity='variance',maxDepth=maxDepth,maxBins=maxBins)
    preds=model.predict(test.map(lambda  x:x.features))
    actual=test.map(lambda x:x.label)
    tp=actual.zip(preds)
    rmsle=np.sqrt(tp.map(lambda (x,y):squared_log_error(x,y)).mean())
    return rmsle


def distinct_data(data,i):
    return np.unique(data[i])

def unique(records):
    unique_key = list(set(records.keys()))
    unique_data = [(unique_key[i], records[unique_key[i]]) for i in range(len(unique_key))]
    data = sc.parallelize(unique_data)
    data = data.map(lambda x: LabeledPoint((x[0]), (x[1])))
    return data


if __name__=="__main__":
    table=connect('192.168.232.69','water_quality')
    row_data = sc.parallelize(get_data(table))
    records = row_data.map(lambda x:((x['info:COND']),(x['info:tem'], x['info:pH'], x['info:DO']))).collectAsMap()
    # 删除重复的元素
    data=unique(records)
    model=train_model(data,20,60)
    print(get_R2(model,data))
    print(get_tp(model,data).take(5))
    # print(np.sqrt(get_tp(model,test).map(lambda (x,y):squared_error(x,y)).mean()))
    # rmsle=np.sqrt(true_vs_preds.map(lambda (x,y):squared_log_error(x,y)).mean())
# 每一次训练得到的均方根对数误差都不一样
