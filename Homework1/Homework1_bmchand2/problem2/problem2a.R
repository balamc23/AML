# Citation http://api.rpubs.com/JanpuHou/304506

setwd('~/Desktop/AML/problem2a')
library(e1071) 
library(caret)

#Gaussian Naive Bayes
library(naivebayes)

#helper function: choosing 0 or 1
thresholdCheck<-function(x){
  if(x < 127){
    x<-0
  } else {
    x<-1
  }
  return(x)
}

train=read.csv('/Users/BalaChandrasekaran/Desktop/AML/train.csv')
test=read.csv('/Users/BalaChandrasekaran/Desktop/AML/test.csv')
train_dataset=as.data.frame(train)
test_dataset= as.data.frame(test)
sapply(train_dataset[1, ], class)
sapply(test_dataset[1, ], class)

# cleaning training data
train_dataset[, 1] <- as.factor(train_dataset[, 1])
colnames(train_dataset) <- c("Y", paste("X.", 1:784, sep = ""))
class(train_dataset[, 1])
levels(train_dataset[, 1])
sapply(train_dataset[1, ], class)

# cleaning for testing data
test_dataset[, 1] <- as.factor(test_dataset[, 1]) # As Category
colnames(test_dataset) <- c("Y", paste("X.", 1:784, sep = "")) ##similar to the training set
class(test_dataset[, 1])
levels(test_dataset[, 1])
sapply(test_dataset[1, ], class)

# training and building Gaussian nb 
process_time = proc.time()
nb_model<-naiveBayes(train_dataset$Y ~., data = train_dataset)
proc.time() - process_time
summary(nb_model)

# test Gaussian nb
nb_accuracy<-predict(nb_model, newdata = test_dataset, type = "class")
confusionMatrix(data = nb_accuracy, test_dataset$Y)

# Bernoulli
train_ <- as.data.frame(train_dataset[,2:785])

# attempted to figure out stretching
stretch_ <- as.data.frame(train_dataset[,2:785])

apply(train_, 1:2, thresholdCheck)
bernoulli_nb <- naive_bayes(x = train_, y = train_dataset$Y)
b_accuracy <- predict(bernoulli_nb, test_dataset)
confusionMatrix(data = b_accuracy, test_dataset$Y)




