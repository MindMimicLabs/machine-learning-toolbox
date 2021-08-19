# DL Structured Regression ----
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
if (!require("quantmod")) {
  install.packages("quantmod")
  library(quantmod)
}
start <- as.Date(Sys.Date()-(365*10))
end <- as.Date(Sys.Date())
getSymbols("GOOG", src = "yahoo", from = start, to = end)
data = data.frame(GOOG)
data = data.frame(data[,-c(4)], GOOG.Close = data[,4])

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

apply(y_train, 2, range)
maxs2 <- apply(y_train, 2, max) 
mins2 <- apply(y_train, 2, min)
scaled2 <- as.data.frame(scale(y_train, center = mins2, scale = maxs2 - mins2))
y_train <- scaled2
y_train = data.matrix(y_train)

model.data = data.frame(x_train, y_train)
model.data = data.frame(model.data)
num.col = ncol(model.data)-1
colnames(model.data) = c(1:num.col, 'target')
str(model.data)

folds <- createFolds(y = model.data[, 'target'], k = 10, list = F)
model.data$folds <- folds

FLAGS <- flags(
  flag_numeric('batch_size', 40),
  flag_numeric('epochs', 10),
  flag_numeric('val_split', 0.20)
)

for(f in unique(model.data$folds)){
  
  cat("\n Fold: ", f)
  ind <- which(model.data$folds == f) 
  train_df <- model.data[-ind, -c(num.col+1, num.col+2)]
  y_train <- as.matrix(model.data[-ind, 'target'])
  #y_train <- to_categorical(y_train)
  valid_df <- as.matrix(model.data[ind, -c(num.col+1, num.col+2)])
  y_valid <- as.matrix(model.data[ind, 'target'])
  #y_valid <- to_categorical(y_valid)
  k = ncol(train_df)
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = 128, activation = 'relu', input_shape = k) %>%
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 64, activation = 'relu')%>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.1) %>% 
    layer_dense(units = 1000, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'linear')
  ## compile the model
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_nadam(),  
    metrics = c('mae')
  )
  
  summary(model)
  
  hist <- model %>%
    fit(
      as.matrix(train_df), y = y_train,
      batch_size = FLAGS$batch_size,
      epochs = FLAGS$epochs,
      validation_split = FLAGS$val_split,
      callbacks = list(callback_early_stopping(patience = 10, monitor = 'val_loss', restore_best_weights = TRUE),
                       callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 1e-1, 
                                                     patience=9, verbose=1, mode='auto', 
                                                     min_delta=1e-1, cooldown=0, min_lr=1e-8))
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
  
  postResample(pred = results$y_pred, obs = results$y_valid)
}

#Save the model
model %>% save_model_hdf5("model7.h5")
load_model_hdf5("model7.h5")
