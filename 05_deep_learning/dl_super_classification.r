# DL Structured Classification ----
if (!require("readr")) {
  install.packages("readr")
  library(readr)
}

if (!require("caret")) {
  install.packages("caret")
  library(caret)
}

if (!require("GGally")) {
  install.packages("GGally")
  library(GGally)
}

if (!require("keras")) {
  install.packages("keras")
  library(keras)
}

if (!require("tensorflow")) {
  install.packages("tensorflow")
  library(tensorflow)
}

if (!require("dummies")) {
  install.packages("dummies")
  library(dummies)
}
PDDM.uci <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"), header=TRUE)
PDDM.uci = PDDM.uci[,-c(1)]
PDDM.uci = data.frame(apply(PDDM.uci, 2, as.numeric))
data = data.frame(PDDM.uci[,-c(17)])
status = as.factor(PDDM.uci$status)



#preProClean <- preProcess(x = data, method = c("scale", "center"))
#data <- predict(preProClean, data %>% na.omit)
data = data.frame(cbind(data, status))


N = nrow(data)
Ind = sample(N, N*1, replace = FALSE) 
p = ncol(data)
y_train = data.matrix(data[Ind, c(p)])
x_train  = data.matrix(data[Ind, -c(p)])

apply(x_train, 2, range)
maxs2 <- apply(x_train, 2, max) 
mins2 <- apply(x_train, 2, min)
scaled2 <- as.data.frame(scale(x_train, center = mins2, scale = maxs2 - mins2))
x_train <- scaled2
x_train = data.matrix(x_train)


model.data = data.frame(x_train, y_train)
model.data = data.frame(model.data)
num.col = ncol(model.data)-1
colnames(model.data) = c(1:num.col, 'label')
str(model.data)

folds <- createFolds(y = model.data[, 'label'], k = 10, list = F)
model.data$folds <- folds

FLAGS <- flags(
  flag_numeric('batch_size', 2),
  flag_numeric('epochs', 10),
  flag_numeric('val_split', 0.20)
)

for(f in unique(model.data$folds)){
  
  cat("\n Fold: ", f)
  ind <- which(model.data$folds == f) 
  train_df <- model.data[-ind, -c(num.col+1, num.col+2)]
  y_train <- as.matrix(model.data[-ind, 'label'])
  y_train <- to_categorical(y_train) # 
  valid_df <- as.matrix(model.data[ind, -c(num.col+1, num.col+2)])
  y_valid <- as.matrix(model.data[ind, 'label'])
  y_valid <- to_categorical(y_valid)
  k = ncol(train_df)
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = 128, activation = 'relu', input_shape = k) %>%
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 64, activation = 'relu')%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 1000, activation = 'relu') %>%
    layer_dense(units = 2, activation = 'sigmoid') # softmax 
  ## compile the model
  model %>% compile(
    loss = 'binary_crossentropy', # categorical_crossentropy 
    optimizer = optimizer_sgd(),  
    metrics = c('accuracy')
  )
  
  summary(model)
  
  hist <- model %>%
    fit(
      as.matrix(train_df), y = y_train,
      batch_size = FLAGS$batch_size,
      epochs = FLAGS$epochs,
      validation_split = FLAGS$val_split,
      callbacks = list(callback_early_stopping(patience = 10, monitor = 'val_loss', restore_best_weights = TRUE))
    )
  
  print(plot(hist))
  
  
  df_out <- hist$metrics %>% 
    {data.frame(acc = .$acc[FLAGS$epochs], val_acc = .$val_acc[FLAGS$epochs])}
  
  print(df_out)
  
  score <- model %>% evaluate(valid_df, y_valid, batch_size = FLAGS$batch_size)
  print(score)
  
  pred <- model %>% predict(valid_df, y_valid, batch_size = FLAGS$batch_size)
  y_pred = round(pred)
  print(head(y_pred))
  
  results = data.frame(y_valid, y_pred)
  print(head(results))
  
  cm = confusionMatrix(as.factor(y_pred), reference = as.factor(y_valid), mode = "prec_recall")
  print(cm)
  
}

#Save the model
model %>% save_model_hdf5("model7.h5")
load_model_hdf5("model7.h5")


fold1 = as.numeric(c(83.16,83.25,83.03,83.14))
fold2 = as.numeric(c(72.76,73.57,71.05,72.29))
fold3 = as.numeric(c(82.69,82.74,82.63,82.69))
fold4 = as.numeric(c(80.46,80.50,80.39,80.45))
fold5 = as.numeric(c(83.82,83.82,83.82,83.82))
fold6 = as.numeric(c(82.11,82.19,81.97,82.08))
fold7 = as.numeric(c(74.54,74.97,73.68,74.32))
fold8 = as.numeric(c(74.54,74.00,75.66,74.82))
fold9 = as.numeric(c(0.8889,0.8889,0.8889,0.8889))
fold10 = as.numeric(c(50.00,50.00,50.00,50.00))
data <- data.frame(rbind(fold1,fold2,fold3,fold4,fold5,
                         fold6,fold7,fold8,fold9,fold10
))
colnames(data) = c('accuracy','precision','recall','F1')

str(data)
