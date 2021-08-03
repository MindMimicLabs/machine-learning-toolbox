# Naive Bayes in R ----
library(caret)
require(naivebayes)
data("iris")
iris = iris
str(iris)
index = createDataPartition(iris[,1], p =0.70, list = FALSE)
training = iris[index,]
valid = iris[-index,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.NB <- train(Species~., data = training, method = "naive_bayes", trControl=control, metric=metric)
fit.NB
data.pred = predict(fit.NB, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
print(cm)