setwd("~/Kaggle/Titanic/outputcsv")
agg<-read.csv("agg.csv")
table (agg$Survived)
agg<-agg[agg$Survived!="Survived",]
agg$Survived<-as.numeric(as.character(agg$Survived))
aggsum<-aggregate(agg$Survived, by=list(agg$PassengerId), FUN=sum)
table(aggsum)
aggsum$x<-aggsum$x/max(aggsum$x)
colnames(aggsum)<-c("PassengerId","Survived")



colnames(aggsum)<-c("PassengerId","Survived")
aggsumbk<-aggsum
for (i in seq(0.4,1, by=.074)){
  aggsum<-aggsumbk
  aggsum$Survived[aggsum$Survived>=i]<-1
  aggsum$Survived[aggsum$Survived!=1]<-0
  write.csv(aggsum, paste0("sensitivty",i,".csv"), row.names=FALSE)
 print(sum(aggsum$Survived>=i)) 
}