# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:52:45 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
csi = pd.read_csv('C:/users/administrator/desktop/csi.csv',header = 0)
ssi = pd.read_csv('C:/users/administrator/desktop/ssi.csv',header = 0)
dic = {}
for i in range(0,csi.shape[0]):
    sum = 0
    for j in range(0,ssi.shape[0]):
        if ssi.iloc[j,0]+str(ssi.iloc[j,1]) == csi.iloc[i,2]:
            sum = sum +ssi.iloc[j,3]
    dic[csi.iloc[i,2]] = sum
dic_df = pd.Series(dic)
#dic_Df = pd.DataFrame(dic_df)
dic_df.sort_values(ascending = False)
ssi.sort_values(by = ['class','course','score'],ascending = (True,True,False))
