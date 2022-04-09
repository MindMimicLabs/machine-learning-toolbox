library(keras)
library(xml2)
library(dplyr)
library(plyr)
library(searcher)
library(tokenizers)
library(tm)
library(stringr)
library(topicmodels)
library(doParallel)
library(ggplot2)
library(scales)
library(qdapDictionaries)
library(data.table)
library(readtext)
library(caret)


data4 = data.frame(read.csv("data.csv"))
data4 = data4[,-1]

## - [3] - Modeling... 


FLAGS <- flags(
  flag_numeric('dropout1', 0.5),
  flag_integer('filters_cnn', 32),
  flag_integer('filters_lstm', 32),
  flag_numeric('reg1', 5e-4),
  flag_numeric('reg2', 0.01),
  flag_numeric('batch_size', 40),
  flag_numeric('maxlen', 2000),
  flag_numeric('max_features', 60000),
  flag_numeric('embedding_dim', 1000),
  flag_numeric('leaky_relu', 0.50),
  flag_numeric('kernel', 5),
  flag_numeric('epochs', 100),
  flag_numeric('pool_size', 4),
  flag_numeric('lr', 0.03),
  flag_numeric('val_split', 0.20)
)

tokenizer <- text_tokenizer(num_words = FLAGS$max_features)

tokenizer %>% 
  fit_text_tokenizer(data4$text)


text_seqs <- texts_to_sequences(tokenizer, data4$text)


train_set4 = text_seqs[1:1200]
valid_set4 = text_seqs[1201:1422]

train_set5 = data4$V2[1:1200]
valid_set5 = data4$V2[1201:1422]

#Set parameters:

x_train <- train_set4 %>%
  pad_sequences(maxlen = FLAGS$maxlen)

x_valid <- valid_set4 %>%
  pad_sequences(maxlen = FLAGS$maxlen)

y_train <- as.matrix(train_set5)
y_valid <- as.matrix(valid_set5)

y_train <- to_categorical(y_train)
y_valid <- to_categorical(y_valid)

model <- keras_model_sequential() %>% 
  layer_embedding(FLAGS$max_features, FLAGS$embedding_dim, input_length = FLAGS$maxlen) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/4, FLAGS$kernel, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_conv_1d(
    FLAGS$filters_cnn, FLAGS$kernel-1, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%  
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/2, FLAGS$kernel-3, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_max_pooling_1d(FLAGS$pool_size) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/2, FLAGS$kernel-3, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/4, FLAGS$kernel-3, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/2, FLAGS$kernel-3, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_conv_1d(
    FLAGS$filters_cnn/4, FLAGS$kernel-3, 
    kernel_initializer = "VarianceScaling",
    kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
    padding = "same", strides = 1L
  ) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_max_pooling_1d(FLAGS$pool_size) %>%
  layer_batch_normalization()  %>%
  layer_dropout(0.5) %>%
  layer_lstm(units=FLAGS$filters_lstm, 
             kernel_initializer = "VarianceScaling",
             kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg2, l2 = FLAGS$reg2)) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_dropout(FLAGS$leaky_relu) %>%
  layer_dense(ncol(y_train)) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_sgd(lr = FLAGS$lr),
    metrics = "accuracy"
  )

hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = FLAGS$batch_size,
    epochs = FLAGS$epochs,
    validation_split = FLAGS$val_split,
    callbacks = list(callback_early_stopping(patience = 2, monitor = 'val_acc', restore_best_weights = TRUE),
                     callback_reduce_lr_on_plateau(monitor = 'val_acc', factor = 1e-1, 
                                                   patience=5, verbose=2, mode='auto', 
                                                   min_delta=1e-1, cooldown=0, min_lr=1e-8))
  )



score <- model %>% evaluate(x_valid, y_valid, batch_size = FLAGS$batch_size)
print(score)

#Save the model
model %>% save_model_hdf5("model7.h5")