setwd('~/Desktop/AML')
rm(list=ls())
data<-read.csv('pima-indians-diabetes.data.csv', header=FALSE)
library(klaR)
library(caret)
bigx<-data[,-c(9)]
bigy<-as.factor(data[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
svm<-svmlight(bigx[wtd,], bigy[wtd], pathsvm='/Users/BalaChandrasekaran/Downloads/svm_light_OS10.8.4_i7')
labels<-predict(svm, bigx[-wtd,])
foo<-labels$class
sum(foo==bigy[-wtd])/(sum(foo==bigy[-wtd])+sum(!(foo==bigy[-wtd])))