# coding = "utf8"
#linear_regression
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression

train_data = pd.read_csv('./Processed_datasets/Datasets_WithGdp.csv',encoding = 'gbk',header = 0,index_col =0)
X_new = SelectKBest(f_regression,k=5).fit_transform(train_data.iloc[:,:-1],train_data.iloc[:,-1])
Y = np.array(train_data.iloc[:,-1])
model = LinearRegression()
model.fit(X_new,Y)
index_fitted = pd.read_csv('./Processed_datasets/Index_ch_fitted.csv',encoding = 'gbk',header = None,index_col = None)
predict_raise_rate = model.predict(np.array(index_fitted))
print(predict_raise_rate)
