# - Support Vector Machines in R ----

# Linear ----
# Kernel = linear
library(caret)
library(quantmod)
start <- as.Date(Sys.Date()-(365*5))
end <- as.Date(Sys.Date())
getSymbols("GOOG", src = "yahoo", from = start, to = end)
data = GOOG
colnames(data) = c("Open", "High", "Low", "Close", "Volume", "Adjusted")
control <- trainControl(method="cv", number=10)
metric <- "Rsquared"
split<-createDataPartition(y = data$Close, p = 0.7, list = FALSE)
train<-data[split,]
valid<-data[-split,]
set.seed(7)
fit.svm <- train(Close~., data = train, method = "svmLinear", trControl=control, metric=metric)
fit.svm
predictedValues<-predict(fit.svm, valid)
modelvalues<-data.frame(obs = valid$Close, pred=predictedValues)
postResample(pred = predictedValues, obs = valid$Close)

# classification ----
# kernel = linear
library(caret)
data("iris")
iris = iris
str(iris)
index = createDataPartition(iris[,1], p =0.70, list = FALSE)
training = iris[index,]
valid = iris[-index,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.svm <- train(Species~., data = training, method = "svmLinear", family=binomial(), trControl=control, metric=metric)
fit.svm
data.pred = predict(fit.svm, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)