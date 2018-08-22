

library(MARSS)


# ### 方案二



data1 = read.csv('C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/20180813/xulianji.csv')



features = data1[,2:20]  
#共计18个指标



#Y = data1[,20]


# ### 预测2017第三季度、2017第四季度、2018第一季度


features1 = t(features)
features1 = features1[,1:34] #2017年第四季度
#features1 = features1[,1:33] #2018第1季度
#features1 = features1[,1:34] #2018第2季度
line1 = rep(NA,19)
features1 = cbind(features1, line1)
#features1
modd = MARSS(features1)
modd1 = MARSS(y=features1,inits = modd$par)
outt = augment.marssMLE(x=modd1,type.predict = c("observations", "states"), interval = "confidence", conf.level = 0.995)
outt
f = c('规模以上工业增加值增速','零售业商品销售额增速','餐饮业营业额增速','住宿业营业额增速','建筑业现价增长速度')
subset(outt, t==35 & .rownames %in% f)
subset(outt, t==35 & .rownames %in% f)$.fitted