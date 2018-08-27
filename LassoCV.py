# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


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
from sklearn.linear_model import LassoCV
lasso = Lasso()

# In[13]:
df_sheet = pd.read_excel('./Index0820.xlsx',
                          sheet_name=0, encoding='utf8',header=0,index_col=0)
# In[16]:
df_sheet = df_sheet.T
# In[19]:
train_data = df_sheet.copy()
# In[20]:
null_1,null_2 = np.where(train_data.isnull());np.where(train_data.isnull())
# In[21]:
ind=np.arange(1,len(null_1)+1)
miss_matrix=np.array([null_1,null_2,ind])
for i in ind:
        #train_data.iloc[i,j] = mean_df[i]
        train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]] = (train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]-1]+train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]+1])/2
# In[23]:
#train_data = train_data.iloc[:,:-3]
train_data = train_data.T
# In[24]:
train_data.to_csv('./train_datasets.csv',encoding="utf_8_sig")
# In[28]:
gdp_data= pd.read_excel('C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/季度.xlsx',sheet_name=0,skiprows=29,usecols=34,encoding='utf8',header=None,index_col=0)
# In[100]:
#选择多少年的数据
#gdp_data = gdp_data.iloc[:,1:-2]
# In[101]:
gdp_data = gdp_data.T
gdp_data.index = train_data.index
# In[38]:
#lasso.fit(train_data,gdp_data)
lassocv = LassoCV()
lassocv = LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,max_iter=2000, n_alphas=100, n_jobs=1, normalize=False, positive=False,precompute='auto', random_state=None, selection='cyclic', tol=0.0001,verbose=False)
lassocv.fit(train_data,gdp_data.values.ravel())
alpha = lassocv.alpha_
lasso =Lasso(alpha)
lasso.fit(train_data,gdp_data.values.ravel())
predict_raise_rate = lasso.predict(np.array([[9.0,11.9,11.1,12.3,20.8,7.2,45.0,11.0,-4.6,12.0,11.0,21.9,8.5,8.9,1.2,13.6,13.0,8.2,11.0,13.6,5.9,4.9,4.9,12.3,16.6]]))
print((train_data.T).index[np.argwhere(lasso.coef_ != 0)])