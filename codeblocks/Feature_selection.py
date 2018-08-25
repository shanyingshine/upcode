# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#读入数据
train_data = pd.read_csv('./Processed_datasets/Datasets_WithGdp.csv',encoding="gbk",index_col = 0)

#fRegression
k=int(input("输入k值"))
feature_select = SelectKBest(f_regression,k).fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])
feature_index   = np.array(train_data.columns[np.argsort(-feature_select.scores_)[:k]])

print(feature_index)

FIndex = pd.DataFrame(feature_index)
FIndex.to_csv('./Processed_datasets/feature_index.csv',index = 0, header = 0, encoding = 'gbk')