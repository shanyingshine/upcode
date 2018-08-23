
# coding: utf-8
#2018年3季度
# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge





df_sheet = pd.read_excel('./Index0820.xlsx',sheet_name=0, encoding='utf8',header=0,index_col=0)




df_sheet = df_sheet.T



#  输入你想预测的季度，然后对数据框进行截取
#ty = input("please input a number")


# In[19]:


train_data = df_sheet.copy()
train_data = train_data.iloc[:,:-1]




null_1,null_2 = np.where(train_data.isnull());np.where(train_data.isnull())





ind=np.arange(1,len(null_1)+1)
miss_matrix=np.array([null_1,null_2,ind])




for i in ind:
        #train_data.iloc[i,j] = mean_df[i]
        train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]] = (train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]-1]+train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]+1])/2





train_data = train_data.T





train_data.to_csv('./训练集.csv',encoding="utf_8_sig")






gdp_data = pd.read_excel('C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/季度.xlsx',sheet_name=0,skiprows=29,usecols=34,encoding='utf8',header=None,index_col=0)
gdp_data = gdp_data.iloc[:,:-1]


gdp_data = gdp_data.T


gdp_data.index = train_data.index






train_data = pd.concat([train_data,gdp_data['地区生产总值']],axis=1)




feature_select = SelectKBest(f_regression,k=8).fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])
feature_index = np.array(train_data.columns[np.argsort(-feature_select.scores_)[:8]])
print(feature_index)
FIndex = pd.DataFrame(feature_index)
FIndex.to_csv('./feature_index.csv',index = 0, header = 0)
X_new = SelectKBest(f_regression,k=8).fit_transform(train_data.iloc[:,:-1],train_data.iloc[:,-1])
Y = np.array(train_data.iloc[:,-1])
model = LinearRegression()
model.fit(X_new,Y)





FIndex = pd.DataFrame(feature_index)





FIndex.to_csv('./feature_index.csv',index = 0, header = 0)





x_2018_1 = np.array([[8.035767,11.380127,17.252605,10.883272,11.578905,9.881143,13.392330,16.019810]])





predict_raise_rate = model.predict(x_2018_1)





print(predict_raise_rate)

clf = Lasso(alpha=0.1)
clf.fit(np.array(train_data.drop(['地区生产总值'],axis=1)),np.array(train_data.iloc[:,-1]))
#RIdge模型训练
Rclf = Ridge(alpha=0.1)
Rclf.fit(np.array(train_data.drop(['地区生产总值'],axis=1)),np.array(train_data.iloc[:,-1]))

#预测2017年3季度的GDP增长速率
predict_raise_rate = clf.predict(np.array([[8.035767,11.380127,17.252605,14.582779,34.750943,21.642574,51.804846,10.883272,33.937977,11.578905,9.881143,22.404905,8.650932,9.953564,2.033537,11.461651,13.392330,9.318323 ,6.902220,13.699665,1.544706,2.572397,-6.196016,12.247019,16.019810]])) # 输入2017年3季度全部特征(排除'工业生产者出厂价格指数')
print("Lasso")
print(predict_raise_rate)
#预测2017年3季度的GDP增长速率
print("岭回归")
predict_raise_rate = Rclf.predict(np.array([[8.035767,11.380127,17.252605,14.582779,34.750943,21.642574,51.804846,10.883272,33.937977,11.578905,9.881143,22.404905,8.650932,9.953564,2.033537,11.461651,13.392330,9.318323 ,6.902220,13.699665,1.544706,2.572397,-6.196016,12.247019,16.019810]])) # 输入2017年3季度全部特征(排除'工业生产者出厂价格指数')
print(predict_raise_rate)