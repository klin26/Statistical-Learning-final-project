dat <- read.csv("C:/Users/stat-pc/陳冠霖/統計學習/final project/diabetes.csv")
summary(dat)
dat$Outcome <- factor(dat$Outcome)
sum(dat$Glucose==0)
sum(dat$BloodPressure==0)
sum(dat$BMI==0)
remove.1 <- which(dat$Glucose==0)
dat <- dat[-remove.1, ]
remove.2 <- which(dat$BloodPressure==0)
dat <- dat[-remove.2, ]
remove.3 <- which(dat$BMI==0)
dat <- dat[-remove.3, ]
sum(dat$Glucose==0)
sum(dat$BloodPressure==0)
sum(dat$BMI==0)
summary(dat)
library(ggplot2)
library(GGally)
ggpairs(dat, aes(color = Outcome, alpha=0.5))

set.seed(628)
train <- sample(1:nrow(dat), round(nrow(dat)*0.7))
test <- (-train)
dat.train <- dat[train, ]
dat.test <- dat[test,]


#performance function
perfor_dummy <- function(pred.prob,pred.class,true.class,method,plot=T){ 
  con <- confusionMatrix(pred.class,true.class,positive = "1") 
  Sensitivity <- con$byClass[1]
  Specificity <- con$byClass[2]
  ACC <- con[[3]][1]
  if(plot==T){
    AUC <- roc.curve(true.class, pred.prob,main = method,plotit = T)$auc
    text(0.7,0.1,paste("AUC = ",round(AUC,4)),cex = 1.5)
  }else{
    AUC <- roc.curve(true.class, pred.prob,main = method,plotit = F)$auc
  }
  return(c(round(Sensitivity,5),round(Specificity,5),round(ACC,5),AUC = round(AUC,5)))
}
perfor_label <- function(pred.prob,pred.class,true.class,method,plot=T){ 
  con <- confusionMatrix(pred.class,true.class)
  Sensitivity <- con$byClass[1]
  Specificity <- con$byClass[2]
  ACC <- con[[3]][1]
  if(plot==T){
    AUC <- roc.curve(true.class, pred.prob,main = method,plotit = T)$auc
    text(0.7,0.1,paste("AUC = ",round(AUC,4)),cex = 1.5)
  }else{
    AUC <- roc.curve(true.class, pred.prob,main = method,plotit = F)$auc
  }
  return(c(round(Sensitivity,5),round(Specificity,5),round(ACC,5),AUC = round(AUC,5)))
}


#logistic
#testing
library(nnet)
logi.fit <- multinom(Outcome~.,data = dat.train)
logi.prob <- predict(logi.fit,dat.test,type = "probs")
logi.class.pred <- predict(logi.fit, dat.test, type = "class")
perfor_dummy(logi.prob,logi.class.pred,factor(dat.test$Outcome),"logistic")

#cv
#logi.fit <- multinom(Outcome~.,data = dat,CV=TRUE)
#logi.prob <- predict(logi.fit,dat,type = "probs")
#logi.class.pred <- predict(logi.fit, dat, type = "class")
#perfor_dummy(logi.prob,logi.class.pred,factor(dat$Outcome),"logistic")



#lda
library(MASS) #ldaqda
library(caret) #confusionmatrix
library(ROSE) #ROCcurve
library(klaR) #partimat
lda.fit <- lda(Outcome ~ .,data=dat.train)
#plot(lda.fit)
#partimat(Outcome ~ .,data=dat.train,method = "lda",plot.matrix = TRUE,imageplot = FALSE)
#training
#lda.pred <- predict(lda.fit,dat.train)
#names(lda.pred)
#lda.class <- lda.pred$class
#mean(lda.class==dat.train$Outcome) #ACC
#confusionMatrix(lda.class,factor(dat.train$Outcome),positive="1")
#oc.curve(dat.train$Outcome,lda.pred$posterior[,2],plotit=T)
#perfor_dummy(lda.pred$posterior[,2],lda.pred$class,factor(dat.train$Outcome),"lda")

#testing
lda.pred <- predict(lda.fit,dat.test)
names(lda.pred)
perfor_dummy(lda.pred$posterior[,2],lda.pred$class,factor(dat.test$Outcome),"LDA")

#cv
#lda.fit <- lda(Outcome ~ .,data=dat,CV=TRUE)
#perfor_dummy(lda.fit$posterior[,2],lda.fit$class,factor(dat$Outcome),"lda")

#qda
qda.fit <- qda(Outcome ~ .,data=dat.train)
#training
#qda.pred <- predict(qda.fit,dat.train)
#names(qda.pred)
#qda.class <- qda.pred$class
#mean(qda.class==dat.train$Outcome) #ACC
#confusionMatrix(qda.class,factor(dat.train$Outcome),positive="1")
#roc.curve(dat.train$Outcome,qda.pred$posterior[,2],plotit=T)
#testing
qda.pred <- predict(qda.fit,dat.test)
names(qda.pred)
qda.class <- qda.pred$class
mean(qda.class==dat.test$Outcome) #ACC
confusionMatrix(qda.class,factor(dat.test$Outcome),positive="1")
roc.curve(dat.test$Outcome,qda.pred$posterior[,2],plotit=T)
perfor_dummy(qda.pred$posterior[,2],qda.pred$class,factor(dat.test$Outcome),"QDA")

#cv
#qda.fit <- qda(Outcome ~ .,data=dat,CV=TRUE)
#perfor_dummy(qda.fit$posterior[,2],qda.fit$class,factor(dat$Outcome),"qda")


#naivebayes
library(e1071) #naivebayes
nb.fit <- naiveBayes(Outcome ~ .,data=dat.train)
#training
#nb.class <- predict(nb.fit,dat.train)
#table(nb.class,dat.train$Outcome)
#mean(nb.class==dat.train$Outcome) #ACC
#nb.post <- predict(nb.fit,dat.train,type="raw")
#confusionMatrix(factor(nb.class),factor(dat.train$Outcome),positive="1")
#roc.curve(dat.train$Outcome,nb.post[,2],plotit = T)
#testing
nb.class <- predict(nb.fit,dat.test)
table(nb.class,dat.test$Outcome)
mean(nb.class==dat.test$Outcome) #ACC
nb.post <- predict(nb.fit,dat.test,type="raw")
confusionMatrix(factor(nb.class),factor(dat.test$Outcome),positive="1")
roc.curve(dat.test$Outcome,nb.post[,2],plotit = T)
perfor_dummy(nb.post[,2],factor(nb.class),factor(dat.test$Outcome),"naiveBayes")

#cv
#nb.fit <- naiveBayes(Outcome ~ .,data=dat,CV=TRUE)
#nb.class <- predict(nb.fit,dat)
#nb.post <- predict(nb.fit,dat,type="raw")
#perfor_dummy(nb.post[,2],factor(nb.class),factor(dat$Outcome),"naiveBayes")



as.data.frame(cbind(rbind('Logistic Regression','LDA','QDA','naiveBayes'),
                    rbind(perfor_dummy(logi.prob,logi.class.pred,factor(dat.test$Outcome),"logistic",plot=F),
                          perfor_dummy(lda.pred$posterior[,2],lda.pred$class,factor(dat.test$Outcome),"LDA",plot=F),
                          perfor_dummy(qda.pred$posterior[,2],qda.pred$class,factor(dat.test$Outcome),"QDA",plot=F),
                          perfor_dummy(nb.post[,2],factor(nb.class),factor(dat.test$Outcome),"naiveBayes",plot=F))))
      