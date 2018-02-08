# Citation https://www.kaggle.com/iammangod96/mnist-digit-recognition-using-simple-randomforest/code

library(randomForest)
library(data.table)
library(imager)

#loading data
train_set<-fread('/Users/BalaChandrasekaran/Desktop/AML/train.csv')
test_set<-fread('/Users/BalaChandrasekaran/Desktop/AML/test.csv')

#random forest
train_set$label<-factor(train_set$label)
rf<-randomForest(data = train_set, label ~ ., ntree = 10, depth = 8)

#pred
prediction<-predict(rf, newdata = test_set)

rf