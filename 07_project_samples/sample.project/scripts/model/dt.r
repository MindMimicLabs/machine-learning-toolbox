# Decision Tree ---- 
index = createDataPartition(data[,1], p =0.80, list = FALSE)
training = data[index,]
valid = data[-index,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.rpart <- train(Species~., data = training, method="rpart", metric=metric, trControl=control)
log_print(fit.rpart)
plot(fit.rpart$finalModel, uniform=TRUE,
     main="Classification Tree")
text(fit.rpart$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
data.pred = predict(fit.rpart, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
log_print(cm)
log_print(paste0("Decision Tree Trained/Tested..."))