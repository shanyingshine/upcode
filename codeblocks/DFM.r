library(MARSS)
library(readxl)

data1 <- read.csv("C:/Users/Administrator/iCloudDrive/Documents/code/upcode/codeblocks/Processed_datasets/train_datasets.csv",header = T,check.names = F)
row.names(data1) <- data1[,1]
data1 <- data1[,-1]
ncols <- ncol(data1)
print("please input the season you want to predict")
print("参照格式2018.1")
ty <-edit()
delta_s <- as.integer(((ty-floor(ty))-(as.numeric(colnames(data1)[1])-floor(as.numeric(colnames(data1)[1]))))*10)
delta_t <-as.integer(4*(floor(ty)-floor(as.numeric(colnames(data1)[1])))+delta_s)
features <- data1[,1:delta_t]
line1 <- rep(NA,nrow(features))
features1 <- cbind(features,line1)
features1 <- as.matrix(features1)
#features1
modd = MARSS(features1)
modd1 = MARSS(y=features1,inits = modd$par)
outt = augment.marssMLE(x=modd1,type.predict = c("observations", "states"), interval = "confidence", conf.level = 0.95)
outt

feature_index <- read.csv('C:/Users/Administrator/iCloudDrive/documents/code/upcode/codeblocks/Processed_datasets/feature_index.csv',fileEncoding = "gbk",header = F,stringsAsFactors = F)
feature_index <- as.vector(unlist(feature_index[1]))
subset(outt, t==(delta_t+1) & .rownames %in% feature_index)
indexf <- subset(outt, t==(delta_t+1) )$.fitted
print(indexf)
index_fitted <- subset(outt, t==(delta_t+1) & .rownames %in% feature_index)$.fitted
print(index_fitted)
write.table(t(index_fitted) , 'C:/Users/Administrator/iCloudDrive/Documents/code/upcode/codeblocks/Processed_datasets/Index_ch_fitted.csv',col.names = F,row.names = F,sep = ',')
#write.table(t(index_fitted) , 'C:/Users/Administrator/iCloudDrive/Documents/code/upcode/codeblocks/Processed_datasets/Index_ch_fitted.txt',col.names = F,row.names = F,sep = ',')