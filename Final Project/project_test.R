library(AppliedPredictiveModeling)
library(corrplot)
library(GGally)
library(leaps)
library(glmnet)
library(plotrix)
library(ggplot2)
library(psych)
library(tree)
library(glmnet)
library(gam)
library(splines)
library(randomForest)
library(caret)
library(dplyr)
library(gbm)
library(xgboost)
library(mlbench)
library(ROSE)
library(MASS)
library(e1071)
library(class)
library(nnet)


## EDA
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
ggpairs(dat, aes(color = Outcome, alpha=0.5))
ggpairs(dat, columns = c(2,9), aes(color = Outcome, alpha=0.5)) ## glucose
ggpairs(dat, columns = c(3,9), aes(color = Outcome, alpha=0.5)) ## blood pressure
ggpairs(dat, columns = c(6,9), aes(color = Outcome, alpha=0.5)) ## BMI
ggpairs(dat, columns = c(8,9), aes(color = Outcome, alpha=0.5)) ## age

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
  con <- confusionMatrix(pred.class,true.class, positive = "1")
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

## logistic

library(nnet)
logi.fit <- multinom(Outcome~.,data = dat.train)
logi.prob <- predict(logi.fit,dat.test,type = "probs")
logi.class.pred <- predict(logi.fit, dat.test, type = "class")
perfor_dummy(logi.prob,logi.class.pred,factor(dat.test$Outcome),"logistic")
#confusionMatrix(logi.class.pred, dat.test$Outcome)
#roc.curve(dat.test$Outcome, logi.prob, plotit=T)
#perfor_dummy(logi.prob, logi.class.pred, dat.test$Outcome, "Logistic regression", plot=T)
#cut
logi.class.pred <- ifelse(logi.prob >= 0.3609467, '1', '0')
logi.class.pred <- factor(logi.class.pred)
perfor_dummy(logi.prob,logi.class.pred,factor(dat.test$Outcome),"logistic")


## LDA

library(MASS) #ldaqda
library(caret) #confusionmatrix
library(ROSE) #ROCcurve
lda.fit <- lda(Outcome ~ ., data=dat.train)
plot(lda.fit)
lda.pred <- predict(lda.fit,dat.test)
names(lda.pred)
perfor_dummy(lda.pred$posterior[,2],lda.pred$class,factor(dat.test$Outcome),"LDA")

#cut
lda.prob <- predict(lda.fit,dat.test,type = "probs")
lda.class.pred <- ifelse(lda.prob$posterior[,2] >= 0.3609467, '1', '0')
lda.class.pred <- factor(lda.class.pred)
perfor_dummy(lda.prob$posterior[,2],lda.class.pred,factor(dat.test$Outcome),"LDA")

## QDA

qda.fit <- qda(Outcome ~ ., data=dat.train)
qda.pred <- predict(qda.fit,dat.test)
names(qda.pred)
qda.class <- qda.pred$class
#mean(qda.class==dat.test$Outcome) #ACC
#confusionMatrix(qda.class,factor(dat.test$Outcome),positive="1")
#roc.curve(dat.test$Outcome,qda.pred$posterior[,2],plotit=T)
perfor_dummy(qda.pred$posterior[,2],qda.pred$class,factor(dat.test$Outcome),"QDA")

#cut
qda.prob <- predict(qda.fit,dat.test,type = "probs")
qda.class.pred <- ifelse(qda.prob$posterior[,2] >= 0.3609467, '1', '0')
qda.class.pred <- factor(qda.class.pred)
perfor_dummy(qda.prob$posterior[,2],qda.class.pred,factor(dat.test$Outcome),"QDA")



## NB

library(e1071) #naivebayes
nb.fit <- naiveBayes(Outcome ~ ., data=dat.train)
nb.class <- predict(nb.fit,dat.test)
#table(nb.class,dat.test$Outcome)
#mean(nb.class==dat.test$Outcome) #ACC
nb.post <- predict(nb.fit,dat.test,type="raw")
#confusionMatrix(factor(nb.class),factor(dat.test$Outcome),positive="1")
#roc.curve(dat.test$Outcome,nb.post[,2],plotit = T)
perfor_dummy(nb.post[,2],factor(nb.class),factor(dat.test$Outcome),"naiveBayes")

#cut
nb.prob <- predict(nb.fit,dat.test,type = "raw")
nb.class.pred <- ifelse(nb.prob[,2] >= 0.3609467, '1', '0')
nb.class.pred <- factor(nb.class.pred)
perfor_dummy(nb.prob[,2],nb.class.pred,factor(dat.test$Outcome),"naiveBayes")



## Tree

set.seed(628)
tree.dia <- tree(Outcome ~ ., dat, subset = train)
treecv.dia <- cv.tree(tree.dia, FUN = prune.tree)
plot(treecv.dia$size, treecv.dia$dev, type = "b")

prune.dia <- prune.tree(tree.dia, best = 4)
plot(prune.dia)
text(prune.dia, pretty = 0)
tree.pred <- predict(prune.dia, dat.test, type = "class")
table(tree.pred, dat.test$Outcome)
confusionMatrix(factor(tree.pred), factor(dat.test$Outcome), positive = "1")
roc.curve(dat.test$Outcome, predict(prune.dia, dat.test)[, 2], plotit = T)
perfor_label(predict(prune.dia, dat.test)[, 2], tree.pred, dat.test$Outcome, "Tree", plot=T)

#cut
tree.prob <- predict(prune.dia, dat.test)
tree.class.pred <- ifelse(tree.prob[,2] >= 0.3609467, '1', '0')
tree.class.pred <- factor(tree.class.pred)
perfor_dummy(tree.prob[,2],tree.class.pred,factor(dat.test$Outcome),"Tree")



## Random forest

k = 5
set.seed (628)
folds=sample(1:k, nrow(dat.train), replace=TRUE)
acc.rf=matrix(NA, k, 8)
bestrf.para <- matrix(NA, 8, 9)
for(c in 1:9){
  for(i in 1:8){
    for(j in 1:k){
    rf.fit = randomForest(Outcome ~ ., data = dat.train[folds != j,], mtry=i, importance = TRUE, ntree = 500 + 500 * c)
    pred = predict(rf.fit, dat.train[folds == j,], type = "class")
    pred <- as.numeric(pred)
    pred[pred == 1] <- 0
    pred[pred == 2] <- 1
    acc.rf[j, i] = mean(pred == dat.train$Outcome[folds == j])
    }
  }
  apply(acc.rf, 2, median)[which.max(apply(acc.rf, 2, median))]
  bestrf.para[which.max(apply(acc.rf, 2, median)), c] <- apply(acc.rf, 2, median)[which.max(apply(acc.rf, 2, median))]
}

saveRDS(bestrf.para,"randomforest.bestparameter.rds")

bestrf.para[is.na(bestrf.para)] <- 0
which(bestrf.para==max(bestrf.para),arr.ind=T)

set.seed(628)
rf.dia <- randomForest(Outcome ~ ., data = dat , subset = train , mtry = 5, importance = TRUE, ntree = 1000)

rf.pred.c <- predict(rf.dia, newdata=dat.test, type = "class")
rf.pred.p <- predict(rf.dia, newdata=dat.test, type = "prob")
table(rf.pred.c, dat.test$Outcome)
confusionMatrix(factor(rf.pred.c), factor(dat.test$Outcome), positive = "1")
roc.curve(dat.test$Outcome, rf.pred.p[, 2], plotit = T)
perfor_label(rf.pred.p[, 2] ,factor(rf.pred.c),factor(dat.test$Outcome),"Random Forest")

#cut
rf.prob <- predict(rf.dia,dat.test,type = "prob")
rf.class.pred <- ifelse(rf.prob[,2] >= 0.3609467, '1', '0')
rf.class.pred <- factor(rf.class.pred)
perfor_dummy(rf.prob[,2],rf.class.pred,factor(dat.test$Outcome),"Random Forest")



## Boosting

k = 5
set.seed(628)
folds=sample(1:k, nrow(dat.train), replace=TRUE)
acc.boo=matrix(NA, k, 35)
bestboo.para <- matrix(NA, 35, 5)
depth <- c(1, 2, 3, 4, 5)
lambda <- c(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
tune <- expand.grid(depth, lambda)
set.seed(628)
for(c in 1:5){
  for(i in 1:35){
    for(j in 1:k){
      boost.fit = gbm(I(Outcome == "1") ~ ., data = dat.train[folds != j,], distribution = "bernoulli", interaction.depth = tune$Var1[i], shrinkage = tune$Var2[i], n.tree = 500 + 500 * c)
      pred = predict(boost.fit, dat.train[folds == j,], type = "response")
      pred <- ifelse(pred>=0.5,1,0)
      acc.boo[j, i] = mean(pred == dat.train$Outcome[folds == j])
    }
  }
  apply(acc.boo, 2, median)[which.max(apply(acc.boo, 2, median))]
  bestboo.para[which.max(apply(acc.boo, 2, median)), c] <- apply(acc.boo, 2, median)[which.max(apply(acc.boo, 2, median))]
}

saveRDS(bestboo.para,"boosting.bestparameter.rds")

bestboo.para[is.na(bestboo.para)] <- 0
which(bestboo.para==max(bestboo.para),arr.ind=T)

set.seed (628)
boo.dia = gbm(I(Outcome == "1") ~ ., data = dat.train[folds != j,], distribution = "bernoulli", interaction.depth = 3, shrinkage = 0.01, n.tree = 1500)

boo.pred <- predict(boo.dia, newdata=dat.test, type = "response")
boo.pred <- ifelse(boo.pred>=0.5, 1, 0)
table(boo.pred, dat.test$Outcome)
confusionMatrix(factor(boo.pred), factor(dat.test$Outcome))
roc.curve(dat.test$Outcome, predict(boo.dia, newdata=dat.test, type = "response"), plotit = T)
perfor_dummy(predict(boo.dia, newdata=dat.test, type = "response"), factor(boo.pred), factor(dat.test$Outcome), "Boosting", plot=T)

#cut
boo.prob <- predict(boo.dia,dat.test,type = "prob")
boo.class.pred <- ifelse(rf.prob[,2] >= 0.3609467, '1', '0')
boo.class.pred <- factor(rf.class.pred)
perfor_dummy(rf.prob[,2],rf.class.pred,factor(dat.test$Outcome),"Random Forest")


## SVM (radial)

set.seed(628)
k = 5
folds=sample(1:k, nrow(dat.train), replace=TRUE)
gamma = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)
cost = c(0.1, 1, 5, 10, 50, 100, 500, 1000)
tune.svm <- expand.grid(cost, gamma)
acc.svm <- matrix(NA, k, 64)
set.seed(628)
for(j in 1:k){
  for(i in 1:64){
    svm.fit = svm(Outcome ~ ., data = dat.train[folds != j,], kernel = "radial", gamma = tune.svm$Var1[i], cost = tune.svm$Var2[i])
    pred = predict(svm.fit, newdata=dat.train[folds == j,])
    acc.svm[j, i] = mean(pred == dat.train$Outcome[folds == j])
  }
}

matplot(cbind(apply(acc.svm, 2, median),
              apply(acc.svm, 2, max) ,
              apply(acc.svm, 2, min)), type = "l", lty = c(1, 2, 2), col = "black",
        xlab = "tune", ylab = "Accuracy")
which.max(apply(acc.svm, 2, median))
abline(v = which.max(apply(acc.svm, 2, median)), col = "red")
legend("topleft", c("median", "min & max"), lty = 1:2)

set.seed (628)
svm.dia = svm(Outcome ~ ., data = dat.train[folds != j,], kernel = "radial", gamma = tune.svm$Var1[which.max(apply(acc.svm, 2, median))], cost = tune.svm$Var2[which.max(apply(acc.svm, 2, median))], probability = TRUE)

svm.pred = predict(svm.dia, newdata = dat.test, probability = TRUE)
table(svm.pred, dat.test$Outcome)
confusionMatrix(factor(svm.pred), factor(dat.test$Outcome))
roc.curve(dat.test$Outcome, attr(svm.pred, "probabilities")[, 2], plotit = T)
perfor_dummy(attr(svm.pred, "probabilities")[, 2], factor(svm.pred), factor(dat.test$Outcome), "SVM ( radial )", plot=T)

## SVM (polynomial)

set.seed(628)
k = 5
folds=sample(1:k, nrow(dat.train), replace=TRUE)
cost.pol = seq(5,10, length=10)
gamma.pol = c(10:20)*0.01
tune.svm.pol <- expand.grid(cost.pol, gamma.pol)
acc.svm.pol <- matrix(NA, k, 110)
set.seed(628)
for(j in 1:k){
  for(i in 1:110){
    svm.fit = svm(Outcome ~ ., data = dat.train[folds != j,], kernel = "polynomial", gamma = tune.svm.pol$Var2[i], cost = tune.svm.pol$Var1[i])
    pred = predict(svm.fit, newdata=dat.train[folds == j,])
    acc.svm.pol[j, i] = mean(pred == dat.train$Outcome[folds == j])
  }
}

matplot(cbind(apply(acc.svm.pol, 2, median),
              apply(acc.svm.pol, 2, max) ,
              apply(acc.svm.pol, 2, min)), type = "l", lty = c(1, 2, 2), col = "black",
        xlab = "tune", ylab = "Accuracy")
which.max(apply(acc.svm.pol, 2, median))
abline(v = which.max(apply(acc.svm.pol, 2, median)), col = "red")
legend("topleft", c("median", "min & max"), lty = 1:2)

set.seed (628)
svm.pol.dia = svm(Outcome ~ ., data = dat.train[folds != j,], kernel = "polynomial", gamma = tune.svm.pol$Var2[which.max(apply(acc.svm, 2, median))], cost = tune.svm.pol$Var1[which.max(apply(acc.svm, 2, median))], probability = TRUE)

svm.pol.pred = predict(svm.pol.dia, newdata = dat.test, probability = TRUE)
table(svm.pol.pred, dat.test$Outcome)
confusionMatrix(factor(svm.pol.pred), factor(dat.test$Outcome), positive = "1")
roc.curve(dat.test$Outcome, attr(svm.pol.pred, "probabilities")[, 2], plotit = T)
perfor_dummy(attr(svm.pol.pred, "probabilities")[, 2], factor(svm.pol.pred), factor(dat.test$Outcome), "SVM ( radial )", plot=T)




# Summary

as.data.frame(cbind(rbind('Logistic Regression','LDA','QDA','naiveBayes','tree','Random Forest','Boosting','SVM ( radial )','SVM ( poly )'),
                    rbind(perfor_dummy(logi.prob,logi.class.pred,factor(dat.test$Outcome),"logistic",plot=F),
                          perfor_dummy(lda.pred$posterior[,2],lda.pred$class,factor(dat.test$Outcome),"LDA",plot=F),
                          perfor_dummy(qda.pred$posterior[,2],qda.pred$class,factor(dat.test$Outcome),"QDA",plot=F),
                          perfor_dummy(nb.post[,2],factor(nb.class),factor(dat.test$Outcome),"naiveBayes",plot=F),
                          perfor_label(predict(prune.dia, dat.test)[, 2], tree.pred, dat.test$Outcome, "Tree", plot=F),
                          perfor_label(rf.pred.p[, 2] ,factor(rf.pred.c),factor(dat.test$Outcome),"Random Forest", plot=F),
                          perfor_dummy(predict(boo.dia, newdata=dat.test, type = "response"), factor(boo.pred), factor(dat.test$Outcome), "Boosting", plot=F),
                          perfor_dummy(attr(svm.pred, "probabilities")[, 2], factor(svm.pred), factor(dat.test$Outcome), "SVM ( radial )", plot=F),
                          perfor_dummy(attr(svm.pol.pred, "probabilities")[, 2], factor(svm.pol.pred), factor(dat.test$Outcome), "SVM ( radial )", plot=F))))
