#######################################################################################
#######################################################################################
#######################################################################################
dir = getwd()
setwd(dir)
#######################################################################################
#######################################################################################
#######################################################################################
index = createDataPartition(prep[,1], p =0.80, list = FALSE)
training = prep[index,]
valid = prep[-index,]
print(paste0("Training Data Dimensions...",dim(training)))
print(paste0("Validation Data Dimensions...",dim(valid)))
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"
set.seed(7)
fit.rpart <- train(x5~., data = training, method="rpart", metric=metric, trControl=control)
fit.rpart
png("pictures/dt_model_plot.png")
print(plot(fit.rpart$finalModel, uniform=TRUE,
           main="Classification Tree"))
text(fit.rpart$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
dev.off()
saveRDS(fit.rpart, "./states/dt_model.rds")
data.pred = predict(fit.rpart, newdata = valid)
cm = confusionMatrix(as.factor(data.pred), reference = as.factor(valid$x5), mode = "prec_recall")
print(cm)