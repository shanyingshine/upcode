library(MARSS)
library(readxl)

# ### 方案二



#data1 <- read_excel("C:/Users/Administrator/Desktop/Index0820.xlsx",n_max = 34)
data1 <- read.csv("C:/Users/Administrator/iCloudDrive/Documents/code/upcode/train_datasets.csv",header = T,nrows = 35 )
ncols <- ncol(data1)
rows <-  nrow(data1)

features = data1[,2:ncols]
#共计25个指标


# ### 预测2017第三季度、2017第四季度、2018第一季度


features1 = t(features)
features1 = features1[,1:(rows)] #2017年第四季度
#features1 = features1[,1:33] #2018第1季度
#features1 = features1[,1:34] #2018第2季度
line1 = rep(NA,ncols-1)
features1 = cbind(features1, line1)
#features1
modd = MARSS(features1)
modd1 = MARSS(y=features1,inits = modd$par)
outt = augment.marssMLE(x=modd1,type.predict = c("observations", "states"), interval = "confidence", conf.level = 0.995)
outt

feature_index <- read.csv('C:/Users/Administrator/iCloudDrive/documents/code/upcode/feature_index.csv',fileEncoding = "UTF-8",header = F,stringsAsFactors = F)
feature_index <- as.vector(unlist(feature_index[1]))

#f = c('规模以上工业增加值增速','零售业商品销售额增速','餐饮业营业额增速','住宿业营业额增速','建筑业现价增长速度')
subset(outt, t==35 & .rownames %in% feature_index)
indexf <- subset(outt, t==35 )$.fitted
print(indexf)
index_fitted <- subset(outt, t==35 & .rownames %in% feature_index)$.fitted
print(index_fitted)
write.csv(index_fitted , 'C:/Users/Administrator/iCloudDrive/Documents/code/upcode/Index_ch_fitted.csv')
