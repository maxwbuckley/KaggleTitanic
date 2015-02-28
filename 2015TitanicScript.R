##Clear R environment.
rm(list=ls(all=TRUE))

###Set working directory and path to input csv files.
setwd("~/Kaggle/Titanic")
rawcsv<-"rawcsv/"
data<-read.csv(paste0(rawcsv,"train.csv"), stringsAsFactors=F)
test<-read.csv(paste0(rawcsv,"test.csv"), stringsAsFactors=F)

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

processData <- function(data){
  data <- enrichData(data)
  data <- refineData(data)  
  
  #pclass  sex and title should be considered as a factors
  data$FamilySize<-data$Parch+data$SibSp
  data$Pclass <- as.factor(data$Pclass)
  data$Sex <- as.factor(data$Sex)
  data$Title <- as.factor(data$Title)
  
  return(data);
}

data<-processData(data)
test<-processData(test)
##Replace NA's with the median age for that title
avgagetitle<-tapply(data$Age, data$Title, median, na.rm=TRUE)

for(j in 1:length(avgagetitle)){
  data$Age[is.na(data$Age) & data$Title==rownames(avgagetitle)[j]]<-avgagetitle[j]
  test$Age[is.na(test$Age) & test$Title==rownames(avgagetitle)[j]]<-avgagetitle[j]  
  data$AgeClass<- data$Age*as.numeric(as.character(data$Pclass))
  test$AgeClass<- test$Age*as.numeric(as.character(test$Pclass))
  
}

##TODO Fix Later
avgclassfare<-tapply(data$Fare, data$Pclass, median, na.rm=TRUE)

test$Fare[is.na(test$Fare)]<-8.05

## Assert all variables are correctly filled in.
stopifnot(sum(sapply(data, function(x)any(is.na(x))))==0)
stopifnot(sum(sapply(test, function(x)any(is.na(x))))==0)
## Data munging complete.
##-----------------------------------------##

## Folder for my processed csv files which I will use in a later script.

processedcsv<-"processedcsv/"
write.csv(data, paste0(processedcsv,"train.csv"), row.names=F)
write.csv(test, paste0(processedcsv,"test.csv"), row.names=F)
