
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


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


train_data = train_data.T


# In[24]:


train_data.to_csv('./训练集.csv',encoding="utf_8_sig")



# In[28]:


gdp_data= pd.read_excel('C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/季度.xlsx',sheet_name=0,skiprows=29,usecols=34,encoding='utf8',header=None,index_col=0)


# In[38]:
gdp_data = gdp_data.T

gdp_data.index = train_data.index



# In[40]:


train_data = pd.concat([train_data,gdp_data['地区生产总值']],axis=1)


# In[42]:


feature_select = SelectKBest(f_regression,k=5).fit(train_data.iloc[:,:-1],train_data.iloc[:,-1])
feature_index = np.array(train_data.columns[np.argsort(-feature_select.scores_)[:5]])
print(feature_index)
X_new = SelectKBest(f_regression,k=5).fit_transform(train_data.iloc[:,:-1],train_data.iloc[:,-1])
Y = np.array(train_data.iloc[:,-1])
model = LinearRegression()
model.fit(X_new,Y)


# In[45]:


FIndex = pd.DataFrame(feature_index)


# In[49]:


FIndex.to_csv('./feature_index.csv',index = 0, header = 0)


# In[50]:


x_2018_3 = np.array([[8.344700677,11.75129658,14.00472216,11.21353993,13.140061]])


# In[51]:


x_2018_3


# In[52]:


predict_raise_rate = model.predict(x_2018_3)


# In[53]:


predict_raise_rate


# In[54]: