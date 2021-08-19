# Random Forest in R ----

# Classification ----
library(caret)
data("iris")
iris = iris
str(iris)
index = createDataPartition(iris[,1], p =0.80, list = FALSE)
training = iris[index,]
valid = iris[-index,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.rf <- train(Species~., data = training, method="rf", metric=metric, trControl=control)
fit.rf
vi = varImp(fit.rf, scale = FALSE)
plot(vi, top = ncol(training)-1)
data.pred = predict(fit.rf, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)