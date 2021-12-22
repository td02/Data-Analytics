#Load data 
data = read.csv("Assignment-2_Data.csv")
summary(data)
str(data)

# remove ages that do not make sense
data <- data[-c(which(data$age <= 0 | data$age>150)),]

# remove na values
bank <- na.omit(data)
bank$y[bank$y == "no"] = 0
bank$y[bank$y == "yes"] = 1
bank$housing[bank$housing == "no"] = 0
bank$housing[bank$housing == "yes"] = 1
bank$loan[bank$loan == "no"] = 0
bank$loan[bank$loan == "yes"] = 1
bank$default[bank$default == "no"] = 0
bank$default[bank$default == "yes"] = 1

bank$housing <- as.integer(bank$housing)
bank$default <- as.integer(bank$default)
bank$loan <- as.integer(bank$loan)
bank$y <- as.integer(bank$y)


# Creating Training and Testing Sets
library(caTools)

# Normalization
library(caret)

str(bank)
#mean and sd of each variable
preproc = preProcess(bank)

#normalize the data
BankNorm = predict(preproc, bank)

# Creating Training and Testing Sets
set.seed(88)
split = sample.split(BankNorm$y, SplitRatio = 0.75)
Train = subset(bank, split==TRUE)
Test = subset(bank, split==FALSE)

# Building a Logistic Regression Model
str(Train)
bankLog = glm(y ~ age+default+balance+housing+loan+contact+day+duration+campaign+pdays+previous , data = Train, family=binomial)
summary(bankLog)

# Evaluating the Model
PredictTest = predict(bankLog, type="response", newdata = Test)
summary(PredictTest)
tbl = table(Test$y, PredictTest > 0.4)
sum(diag(tbl))/sum(tbl)

library(ROCR)

ROCRpred = prediction(PredictTest, Test$y)
ROCCurve = performance(ROCRpred, "tpr", "fpr")
plot(ROCCurve)
plot(ROCCurve, colorize=TRUE, print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,0.7))
as.numeric(performance(ROCRpred, "auc")@y.values) # AUC value

#Using CART 

#Loading Libraries

library(caTools)
library(rpart)
library(rpart.plot)

#Set baseline
nrow(Train)
sum(Train$y)
sum(Train$y)/nrow(Train)

#CART Analysis
Tree_min50= rpart(y~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome, method = "class", data=Train, minbucket=50)
prp(Tree_min50)
Tree_min50Predict = predict(Tree_min50, newdata = Test, type="class")
tbl_min50 = table(Test$y, Tree_min50Predict)
tbl_min50
sum(diag(tbl_min50))/sum(tbl_min50)
Tree_min100= rpart(y~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome, method = "class", data=Train, minbucket=100)
prp(Tree_min100)
Tree_min100Predict = predict(Tree_min100, newdata = Test, type="class")
tbl_min100 = table(Test$y, Tree_min100Predict)
tbl_min100
sum(diag(tbl_min100))/sum(tbl_min100)
Tree_min200= rpart(y~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome, method = "class", data=Train, minbucket=200)
prp(Tree_min200)
Tree_min200Predict = predict(Tree_min200, newdata = Test, type="class")
tbl_min200 = table(Test$y, Tree_min200Predict)
tbl_min200
sum(diag(tbl_min200))/sum(tbl_min200)
Tree_min75= rpart(y~ age + job + marital + education + default + balance + housing + loan + contact + day + month + duration + campaign + pdays + previous + poutcome, method = "class", data=Train, minbucket=75)
prp(Tree_min75)
Tree_min75Predict = predict(Tree_min75, newdata = Test, type="class")
tbl_min75 = table(Test$y, Tree_min75Predict)
tbl_min75
sum(diag(tbl_min75))/sum(tbl_min75)

#Random Forest
library(randomForest)
Train$y = as.factor(Train$y)
Test$y = as.factor(Test$y)
rfmdel <-randomForest(y ~age+default+balance+housing+loan+contact+day+duration+campaign+pdays+previous+marital+education+poutcome+job+month,data = Train)
varImpPlot(rfmdel) #To Check Variable Importance
prediction_rf<-predict(rfmdel,Test
prediction_rf
#Confusion matrix to validate it
conf_mat<-confusionMatrix(prediction_rf,Test$y)
#Table for Validation
rftable = table(Test$y,prediction_rf)
Accuracy = sum(diag(rftable))/sum(rftable)
Accuracy
ROCRpredrf = as.numeric(predict(rfmdel,Test,type="response"))
ROCrf = prediction(ROCRpredrf,Test$y)
ROCCurverf = performance(ROCrf,measure = "tpr", x.measure = "fpr")
plot(ROCCurverf)
plot(ROCCurverf, colorize=TRUE, print.cutoffs.at=seq(0,1,0.1), text.adj=c(-0.2,0.7))
AUC_rf <- as.numeric(performance(ROCrf, "auc")@y.values) # AUC value
AUC_rf
