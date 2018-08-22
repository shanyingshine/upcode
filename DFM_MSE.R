# Calculate MSE

library(MARSS)
library(readxl)
data <- read_excel("C:/Users/Administrator/Desktop/Index0820.xlsx",n_max = 34)

# 导入数据，行是时间（季度频率），列是影响gdp增速的协变量指标和gdp增速
#data = read.csv('C:/Users/Administrator/Desktop/Index0820.csv',header=F,sep = ",",nrows = 34,fileEncoding = "UTF-8")

cols = ncol(data);cols
fcf <- c()
scf <- c()
tcf <- c()
gdpf <- c()
features = data[,2:cols]
#下面两行是数据中有gdp在最后一行的时候使用
#features = data[,2:(cols-1)]
#Y = data1[,cols]

features1 = t(features)
# 16.4-28
# 17.1-29
# 17.2-30
# 17.3-31
# 17.4-32
# 18.1-33
# 18.2-34
data_sc_gdp = read.csv('C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/20180821/三产数据.csv', header = FALSE)

fc <- data_sc_gdp[1,94:99]

sc <- data_sc_gdp[2,94:99]
 
tc <- data_sc_gdp[3,94:99]

gdp <- data_sc_gdp[4,94:99]


for (ty  in c(28:33))            #ty for test years
{
  features1 = features1[,1:ty] #17年4季度

#nu <- scan(what = "numeric",nmax = 1)

#features1 = features1[,1:nu] 

#features1 = features1[,1:34] #17年4季度

rows = nrow(features1);

#填补空缺值
for(i in 1:(cols-1)){
  for (j in 1:rows){
    if(is.na(features1[i,j])){
      features1[i,j] <- (features1[i,j-1]+features1[i,j+1])/2
    }
  }
}

write.csv(features1,"C:/Users/Administrator/Desktop/data.csv")

cols1 = ncol(features1);cols1

line = rep(NA,rows)

features1 = cbind(features1, line)

features1

modd = MARSS(features1)

modd1 = MARSS(y=features1,inits = modd$par)

outt = augment.marssMLE(x=modd1,type.predict = c("observations", "states"), interval = "confidence", conf.level = 0.95)

outt

subdata = subset(outt, t==cols1);subdata

write.csv(subdata, 'C:/Users/Administrator/iCloudDrive/Documents/杂七杂八/研一/第二学期/统计局/20180821/影响gdp增速协变量预测数据.csv')

data_sc = data_sc_gdp[1:4,66:(ncol(data_sc_gdp)-34+ty)]

line1 = rep(NA,4)

data_sc1 = cbind(data_sc,line1)

data_sc1 = as.matrix(data_sc1)

mod = MARSS(data_sc1)

mod1 = MARSS(y=data_sc1,inits = mod$par)

out_sc = augment.marssMLE(x=mod1,type.predict = c("observations", "states"), interval = "confidence", conf.level = 0.95)

out_sc

subset(out_sc, t==(ncol(data_sc1)))


#fc <- out_sc$y[1:ty]
fcf <- cbind(fcf,out_sc$.fitted[ty+1])
scf <- cbind(scf,out_sc$.fitted[2*ty+2])
tcf <- cbind(tcf,out_sc$.fitted[3*ty+3])
gdpf <- cbind(gdpf,out_sc$.fitted[4*ty+4])


}

x <- c(1:6)
#fc
plot(fc,type = "l",lty=1,col="blue",xlab="2010-2018年各季度第一产业预测效果图",ylab="增速值")
MSE1<-round(sum((fcf-fc)^2)/6,5)
mtext(paste("MSE=", MSE1,""), cex=1.15, line=1)
lines(x,fcf,type = "l",col="red",lty=2)
legend("topright",lty=c(2,1),c("预测值","真实值"),col=c("red","blue"))

#sc
plot(x,sc,type = "l",lty=1,col="blue",xlab="2010-2018年各季度第二产业预测效果图",ylab="增速值")
MSE2<-round(sum((scf-sc)^2)/6,5)
mtext(paste("MSE=", MSE2,""), cex=1.15, line=1)
lines(x,scf,type = "l",col="red",lty=2)
legend("topright",lty=c(2,1),c("预测值","真实值"),col=c("red","blue"))
#tc
plot(x,tc,type = "l",lty=1,col="blue",xlab="2010-2018年各季度第三产业预测效果图",ylab="增速值")
MSE3<-round(sum((tcf-tc)^2)/6,5)
mtext(paste("MSE=", MSE3,""), cex=1.15, line=1)
lines(x,tcf,type = "l",col="red",lty=2)
legend("topright",lty=c(2,1),c("预测值","真实值"),col=c("red","blue"))

#gdp

plot(x,gdp,type = "l",lty=1,col="blue",xlab="2010-2018年各季度gdp预测效果图",ylab="增速值")
MSE4<-round(sum((gdpf-gdp)^2)/6,5)
mtext(paste("MSE=", MSE4,""), cex=1.15, line=1)
lines(x,gdpf,type = "l",col="red",lty=2)
legend("topright",lty=c(2,1),c("预测值","真实值"),col=c("red","blue"))