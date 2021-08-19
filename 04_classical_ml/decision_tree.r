# Decision Tree - Classification ----
library(caret)
data("iris")
iris = iris
str(iris)
index = createDataPartition(iris[,1], p =0.80, list = FALSE)
training = iris[index,]
valid = iris[-index,]
dim(training)
dim(valid)
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.rpart <- train(Species~., data = training, method="rpart", metric=metric, trControl=control)
fit.rpart
plot(fit.rpart$finalModel, uniform=TRUE,
     main="Classification Tree")
text(fit.rpart$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
data.pred = predict(fit.rpart, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)