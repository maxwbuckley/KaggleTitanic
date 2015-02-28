## rm(list=ls(all=TRUE))
library(randomForest)

setwd("~/R/Titanic")
data<-read.csv("train.csv", stringsAsFactors=F)
test<-read.csv("test.csv", stringsAsFactors=F)

enrichData = function(x){
  nameSplit = sapply(x["Name"], function(y) {strsplit(y, "[,.] | [ ]")})
  x["LastName"] = sapply(nameSplit, function(y) {y[1]})
  x["Title"] = sapply(nameSplit, function(y) {y[2]})
  x["OtherName"] = sapply(nameSplit, function(y) {y[3]})    
  
  return(x)
}

refineData = function(x)
{
  ladies = c( "Mme" , "the Countess" , "Lady" ,  "Dona", "Dr")
  x$Title[x$Title %in% ladies & x$Sex=="female"] = "Mrs"
  
  miss = c("Mlle", "Ms")
  x$Title[x$Title %in% miss] = "Miss"
  
  ranks = c("Capt", "Col", "Major","Sir", "Rev", "Don","Dr")
  x$Title[x$Title %in% ranks & x$Sex=="male"] = "Mr"
  
  masters = c("Jonkheer")
  x$Title[x$Title %in% masters] = "Master"
  return(x)
}  
#data = enrichData(data)
##data = refineData(data)

processData <- function(data){
  data <- enrichData(data)
  data <- refineData(data)  
  
  #pclass  sex and title should be considered as a factors
  data$FamilySize<-data$Parch+data$SibSp
  data$Pclass <- as.factor(data$Pclass )
  data$Sex <- as.factor(data$Sex )
  data$Title <- as.factor(data$Title )
  
  return(data);
}
data<-processData(data)
test<-processData(test)
avgagetitle<-tapply(data$Age, data$Title, median, na.rm=TRUE)

for(j in 1:length(avgagetitle)){
  data$Age[is.na(data$Age) & data$Title==rownames(avgagetitle)[j]]<-avgagetitle[j]
  test$Age[is.na(test$Age) & test$Title==rownames(avgagetitle)[j]]<-avgagetitle[j]  
  data$AgeClass<- data$Age*as.numeric(as.character(data$Pclass))
  test$AgeClass<- test$Age*as.numeric(as.character(test$Pclass))
  
}

##Assert all variables are correctly filled in.
stopifnot(sum(sapply(data, function(x)any(is.na(x))))==0)
stopifnot(sum(sapply(test, function(x)any(is.na(x)))==0)
## Data munging complete.
##-----------------------------------------##
setseed(2117)
traintestsplit<-rbinom(nrow(data),1, prob=.20)

trainingset<-data[traintestsplit==0,]
testingset<-data[traintestsplit==1,]

library("caret")
library("ada")

rf<-randomForest(as.factor(Survived)~Sex+Age+Pclass+FamilySize+AgeClass, data=trainingset, ntree=100)
confusionMatrix(table(predict(rf, newdata=testingset),testingset$Survived))


logregplus<-glm(Survived~Title+Age+Embarked+Pclass+SibSp+Parch, family=binomial,data=trainingset)
summary(logregplus)
confusionMatrix(table(round(predict(logregplus, newdata=testingset, type="response")),testingset$Survived))


adamodel<-ada(trainingset$Survived~Title+Age+Pclass+SibSp+Parch,  data=trainingset,iter=100,nu=1,type="discrete")
confusionMatrix(table(predict(adamodel, newdata=testingset),testingset$Survived))

out<-predict(rf, newdata=test)
output<-cbind(test$PassengerId,out)
output[,2]<-output[,2]-1
colnames(output)<-c("PassengerId","Survived")
write.csv(output,"2015rfoutput.csv",row.names=FALSE)

writeWinner<-function(model){
  predic
  
  
}
