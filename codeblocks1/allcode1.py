import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
import rpy2.robjects as robjects
# In[1]:
#如果数据没有列名，即年份，则
#df_sheet = pd.read_csv('./index2.0.csv ',encoding = 'gbk',header = None,index_col = 0 )
#df_sheet =df_sheet.T
##行数是df_sheet.shape[0]，列数是df_sheet.shape[1]
#a = ['2010','2011','2012','2013','2014','2015','2016','2017','2018']
#b = ['.1','.2','.3','.4']
#ab = []
#for i in a:
#    for j in b:
#        s = i + j
#        ab.append(s)
#ab = list(map(float,ab))
#ab=pd.DataFrame(ab)
#for i in range(df_sheet.shape[1]+1,36):
#    ab = ab.drop(i)
#df_sheet = df_sheet.set_index(list(np.array(ab.T)))
# In[2]
#数据集录入,如果有列名
#usecol = int(input("输入训练集个数:"))
df_sheet = pd.read_excel('./Index0825.xlsx',sheet_name=0,encoding='utf8',header=0,index_col=0)
df_sheet = df_sheet.T
#df_sheet = df_sheet.drop(df_sheet.columns[32],axis =1)
df_cols = df_sheet.shape[1]

# In[3]
train_data = df_sheet.copy()

#检查是否有缺失值,并进行填补
null_1,null_2 = np.where(train_data.isnull());np.where(train_data.isnull())
ind=np.arange(1,len(null_1)+1)
miss_matrix=np.array([null_1,null_2,ind])
for i in ind:
        train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]] = (train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]-1]+train_data.iloc[miss_matrix[0,i-1],miss_matrix[1,i-1]+1])/2

train_data.to_csv('./Processed_datasets/train_datasets.csv',encoding="gbk")


train_data = train_data.T
#读入gdp数据
gdp_data= pd.read_excel('./季度.xlsx',sheet_name=0,skiprows=29,encoding='utf8',header=None,index_col=0)
gdp_cols = gdp_data.shape[1]
if (df_cols-gdp_cols)!=0:
    gdp_data = gdp_data.iloc[:,:(df_cols-gdp_cols)]
gdp_data = gdp_data.T


#合并指标数据与gdp数据
gdp_data.index = train_data.index
train_data_withGDP = pd.concat([train_data,gdp_data['地区生产总值']],axis=1)



#读取三产数据
sc_data = pd.read_csv("./sanchan.csv",encoding = "gbk",header = None,index_col = 0, nrows = 3)
if (df_cols-gdp_cols)!=0:
    sc_data = sc_data.iloc[:,64:(df_cols-gdp_cols)]
else:
    sc_data = sc_data.iloc[:,64:]
sc_data = sc_data.T
sc_data.index = train_data.index
train_data_withSC = pd.concat([train_data,sc_data],axis=1)
train_data_withSCGDP = pd.concat([train_data_withSC,gdp_data],axis =1)



#数据集转置
train_data = train_data.T
gdp_data = gdp_data.T
sc_data = sc_data.T
train_data_withSC = train_data_withSC.T
train_data_withGDP =train_data_withGDP.T
train_data_withSCGDP = train_data_withSCGDP.T
#输出gdp，三产和季度指标数据，第一列为指标
for ty in [ 2017.3,2017.4,2018.1,2018.2]:
    print(ty)
    sc_data.to_csv("./Processed_datasets/Sc_data.csv",encoding = 'gbk')
    gdp_data.to_csv("./Processed_datasets/Gdp_data.csv",encoding="gbk")
    train_data_withGDP.to_csv('./Processed_datasets/Datasets_WithGdp.csv',encoding="gbk")
    train_data_withSC.to_csv('./Processed_datasets/Datasets_WithSC.csv',encoding="gbk")
    train_data_withSCGDP.to_csv('./Processed_datasets/Datasets_WithSCGDP.csv',encoding="gbk")
    
    
    #Feature_selection.py
    FS_data = pd.read_csv('./Processed_datasets/Datasets_WithSCGDP.csv',encoding="gbk",index_col = 0,header= 0)
    #ty = float(input("请输入你想预测的年份季度，形如 2018.1: "))

    delta_t= int((10*ty-20101)//10)*4+int((10*ty-20101)%10)
    if delta_t !=33:
        FS_data=FS_data.drop(FS_data.columns[delta_t:],axis =1)
    #FS_data.to_csv("./Processed_datasets/Datasets_WithSCGDP.csv",encoding = 'gbk')
    #fRegression
    delta_t = robjects.IntVector([delta_t])
    robjects.globalenv['delta_t'] = delta_t
    FS_data = FS_data.T
    #k=int(input("输入k值"))
    for k in range(11,15):
    #for i in range(1,4):
    #    feature_select = SelectKBest(f_regression,k).fit(train_data.iloc[:,:-3],train_data.iloc[:,-i])
    #    feature_index   = np.array(train_data.columns[np.argsort(-feature_select.scores_)[:k]])
    #    print(feature_index)
        
        feature_select = SelectKBest(f_regression,k).fit(FS_data.iloc[:,:-1],FS_data.iloc[:,-1])
        feature_index   = np.array(FS_data.columns[np.argsort(-feature_select.scores_)[:k]])
        FIndex = pd.DataFrame(feature_index)
        FIndex.to_csv('./Processed_datasets/feature_index.csv',index = 0, header = 0, encoding = 'gbk')
        #print(FIndex)
        robjects.r.source('./DFM.r')
    
    
    
        #linear_regression
        linear_data = FS_data
        #linear_data = pd.read_csv('./Processed_datasets/Datasets_WithSCGDP.csv',encoding = 'gbk',header = 0,index_col =0)
        #train_data = train_data.T
        X_new = SelectKBest(f_regression,k).fit_transform(linear_data.iloc[:,:-1],linear_data.iloc[:,-1])
        Y = np.array(linear_data.iloc[:,-1])
        model = LinearRegression()
        model.fit(X_new,Y)
        index_fitted = pd.read_csv('./Processed_datasets/Index_ch_fitted.csv',encoding = 'gbk',header = None,index_col = None)
        predict_raise_rate = model.predict(np.array(index_fitted))
        print("  k = %d"  % k)
        print(np.array(FIndex.T))
        print("  时间=%.1f" % ty)
        print("  gdp= %.3f " % predict_raise_rate)