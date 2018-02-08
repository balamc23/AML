setwd('~/Desktop/AML')
data<-read.csv('pima-indians-diabetes.data.csv', header=FALSE)
library(klaR)
library(caret)

print('Problem 1a')
attributes<-data[,-c(9)]
class_values<-data[,9]
training_score<-array(dim=10)
test_score<-array(dim=10)
wtd<-createDataPartition(y=class_values, p=.8, list=FALSE)
nbx<-attributes
ntrbx<-nbx[wtd, ]
ntrby<-class_values[wtd]
trposflag<-ntrby>0
ptregs<-ntrbx[trposflag, ]
ntregs<-ntrbx[!trposflag,]
ntebx<-nbx[-wtd, ]
nteby<-class_values[-wtd]
ptrmean<-sapply(ptregs, mean, na.rm=TRUE)
ntrmean<-sapply(ntregs, mean, na.rm=TRUE)
ptrsd<-sapply(ptregs, sd, na.rm=TRUE)
ntrsd<-sapply(ntregs, sd, na.rm=TRUE)
ptroffsets<-t(t(ntrbx)-ptrmean)
ptrscales<-t(t(ptroffsets)/ptrsd)
ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
ntroffsets<-t(t(ntrbx)-ntrmean)
ntrscales<-t(t(ntroffsets)/ntrsd)
ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
lvwtr<-ptrlogs>ntrlogs
gotrighttr<-lvwtr==ntrby
training_score<-sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr))
pteoffsets<-t(t(ntebx)-ptrmean)
ptescales<-t(t(pteoffsets)/ptrsd)
ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
nteoffsets<-t(t(ntebx)-ntrmean)
ntescales<-t(t(nteoffsets)/ntrsd)
ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
lvwte<-ptelogs>ntelogs
gotright<-lvwte==nteby
test_score<-sum(gotright)/(sum(gotright)+sum(!gotright))

# print(training_score)
print(test_score)


