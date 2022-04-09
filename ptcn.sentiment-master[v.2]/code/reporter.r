# - {} - Visualize & Capture Training Reports for each Models Training Sessions==================================
setwd("C:/Users/jonat/Desktop/project.pcvn")
library(tfruns)
library(keras)

training_run("data.processor.r")

#Model_Reports...

# Hyper Tuned Dl Modeling....
par <- list(
  dropout1= c(0.50),
  filters_cnn= c(32),
  filters_lstm= c(32),
  reg1= c(5e-4),
  reg2= c(5e-4),
  batch_size= c(20),
  maxlen= c(1000),
  max_features = c(60000),
  embedding_dim= c(200),
  leaky_relu= c(0.50),
  kernel= c(5),
  epochs= c(100),
  pool_size= c(4,8),
  lr= c(0.01),
  val_split= c(0.10))

runs <- tuning_run("model2.r", runs_dir = '_tuning', sample = 1, 
                   flags = par)

ls_runs(order = eval_acc, decreasing = T, runs_dir = '_tuning')

best_run <- ls_runs(order = eval_acc, decreasing= T, runs_dir = '_tuning')[1,]

run <- training_run("model3.r",flags = list(
  dropout1 = best_run$flag_dropout1,
  filters_cnn = best_run$flag_filters_cnn,
  filters_lstm = best_run$flag_filters_lstm,
  reg1 = best_run$flag_reg1,
  reg2 = best_run$flag_reg2,
  batch_size = best_run$flag_batch_size,
  maxlen = best_run$flag_maxlen,
  max_features  = best_run$flag_max_features,
  embedding_dim = best_run$flag_embedding_dim,
  leaky_relu = best_run$flag_leaky_relu,
  kernel = best_run$flag_kernel,
  epochs = best_run$flag_epochs,
  pool_size = best_run$flag_pool_size,
  lr = best_run$flag_lr,
  val_split = best_run$flag_val_split))


view_run(run)

