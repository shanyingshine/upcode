# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import rpy2.robjects as robjects
#import os
#os.environ['R_HOME'] = '/path/to/R'
#读入数据
FS_data = pd.read_csv('./Processed_datasets/Datasets_WithSCGDP.csv',encoding="gbk",index_col = 0,header= 0)
ty = float(input("请输入你想预测的年份季度，形如 2018.1: "))
delta_t= int((10*ty-20101)//10)*4+int((10*ty-20101)%10)
if delta_t !=33:
    FS_data=FS_data.drop(FS_data.columns[delta_t:],axis =1)
FS_data.to_csv("./Processed_datasets/Datasets_WithSCGDP.csv",encoding = 'gbk')
#fRegression
delta_t = robjects.IntVector([delta_t])
robjects.globalenv['delta_t'] = delta_t 
k=int(input("输入k值"))
#for i in range(1,4):
#    feature_select = SelectKBest(f_regression,k).fit(train_data.iloc[:,:-3],train_data.iloc[:,-i])
#    feature_index   = np.array(train_data.columns[np.argsort(-feature_select.scores_)[:k]])
#    print(feature_index)
FS_data = FS_data.T
feature_select = SelectKBest(f_regression,k).fit(FS_data.iloc[:,:-1],FS_data.iloc[:,-1])
feature_index   = np.array(FS_data.columns[np.argsort(-feature_select.scores_)[:k]])
FIndex = pd.DataFrame(feature_index)
FIndex.to_csv('./Processed_datasets/feature_index.csv',index = 0, header = 0, encoding = 'gbk')
robjects.r.source('./DFM.r')