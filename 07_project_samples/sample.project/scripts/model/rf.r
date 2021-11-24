# Random Forest ----
index = createDataPartition(data[,1], p =0.80, list = FALSE)
training = data[index,]
valid = data[-index,]
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.rf <- train(Species~., data = training, method="rf", metric=metric, trControl=control)
log_print(fit.rf)
vi = varImp(fit.rf, scale = FALSE)
plot(vi, top = ncol(training)-1)
data.pred = predict(fit.rf, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$Species), mode = "prec_recall")
log_print(cm)
log_print(paste0("Random Forest Trained/Tested..."))