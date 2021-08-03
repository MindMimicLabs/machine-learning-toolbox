# DL Text Generation ----
library(reticulate)
library(keras)
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tokenizers)
library(tm)
library(data.table)
library(harrypotter) 
# Parameters --------------------------------------------------------------
#Set parameters:
FLAGS <- flags(
  flag_integer('filters_cnn', 32),
  flag_integer('filters_lstm', 32),
  flag_numeric('batch_size',40),
  flag_numeric('maxlen', 100),
  flag_numeric('steps', 5),
  flag_numeric('embedding_dim', 3000),
  flag_numeric('kernel', 5),
  flag_numeric('leaky_relu', 0.50),
  flag_numeric('epochs', 100),
  flag_numeric('lr', 0.003)
)

# - [3] - Harry Potter Text ----
list_of_txt <- list.files(path = "data/hp_txt/.", recursive = TRUE,
                          pattern = "\\.txt$", 
                          full.names = TRUE)

for(i in seq_along(length(list_of_txt))) {
  text = lapply(list_of_txt, readtext::readtext)
}
for(i in seq_along(length(text))) {
  string = rbindlist(text)
}
str(string)

# - [4] - Pre-process Text ----
data.text = string$text
data.text = tolower(data.text)
data.text = iconv(data.text, "latin1", "ASCII", sub = " ")
data.text = tm::removeWords(data.text, stopwords("SMART"))
data.text = gsub("^NA| NA ", " ", data.text)
data.text = tm::removeNumbers(data.text)
data.text = tm::stripWhitespace(data.text)
str(data.text)
data.text[1]

# - [5] - Prepare Tokenized Forms of Each Text ----
text =  tokenize_regex(data.text, simplify = TRUE)
print(sprintf("corpus length: %d", length(text)))

vocab <- gsub("\\s", "", unlist(text)) %>%
  unique() %>%
  sort()
print(sprintf("total words: %d", length(vocab))) 

sentence <- list()
next_word <- list()
list_words <- data.frame(word = unlist(text), stringsAsFactors = F)
j <- 1

for (i in seq(1, length(list_words$word) - FLAGS$maxlen - 1, by = FLAGS$steps)){
  sentence[[j]] <- as.character(list_words$word[i:(i+FLAGS$maxlen-1)])
  next_word[[j]] <- as.character(list_words$word[i+FLAGS$maxlen])
  j <- j + 1
}
# Model Definition --------------------------------------------------------


model <- keras_model_sequential() %>% 
  layer_conv_1d(input_shape = c(FLAGS$maxlen, length(vocab)),
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$reg1, l2 = FLAGS$reg1),
                padding = "same", strides = 1L
  ) %>%
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_conv_1d(
    FLAGS$filters_cnn, FLAGS$kernel-1, 
    padding = "same", strides = 1L
  ) %>%  
  layer_activation_leaky_relu(FLAGS$leaky_relu) %>%
  layer_batch_normalization()  %>%
  layer_dropout(0.5) %>%
  layer_lstm(FLAGS$filters_lstm, input_shape = c(FLAGS$maxlen, length(vocab))) %>%
  layer_dropout(FLAGS$leaky_relu) %>%
  layer_dense(length(vocab)) %>%
  layer_activation("softmax")

optimizer <- optimizer_nadam(lr = FLAGS$lr)

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)

# Training & Results ----------------------------------------------------

sample_mod <- function(preds, temperature = 1){
  preds <- log(preds)/temperature
  exp_preds <- exp(preds)
  preds <- exp_preds/sum(exp(preds))
  
  rmultinom(1, 1, preds) %>% 
    as.integer() %>%
    which.max()
}


batch_size <- FLAGS$batch_size
all_samples <- 1:length(sentence)
num_steps <- trunc(length(sentence)/batch_size)

sampling_generator <- function(){
  
  function(){
    
    batch <- sample(all_samples, FLAGS$batch_size)
    all_samples <- all_samples[-batch]
    
    sentences <- sentence[batch]
    next_words <- next_word[batch]
    
    # vectorization
    X <- array(0, dim = c(FLAGS$batch_size, FLAGS$maxlen, length(vocab)))
    y <- array(0, dim = c(FLAGS$batch_size, length(vocab)))
    
    
    for(i in 1:batch_size){
      
      X[i,,] <- sapply(vocab, function(x){
        as.integer(x == sentences[i])
      })
      
      y[i,] <- as.integer(vocab == next_words[i])
      
    }
    
    # return data
    list(X, y)
  }
}

for(i in range(1)){
  model %>% fit_generator(generator = sampling_generator(),
                          steps_per_epoch = num_steps,
                          epochs = FLAGS$epochs,
                          view_metrics = getOption("keras.view_metrics",
                                                   default = "auto"))
}

for(diversity in c(0.2, 0.5, 1, 1.2)){
  
  cat(sprintf("diversity: %f ---------------\n\n", diversity))
  
  start_index <- sample(1:(length(text) - FLAGS$maxlen), size = 1)
  sentence <- text[start_index:(start_index + FLAGS$maxlen - 1)]
  generated <- ""
  
  for(i in 1:200){
    
    x <- sapply(vocab, function(x){
      as.integer(x == sentence)
    })
    x <- array_reshape(x, c(1, dim(x)))
    
    preds <- predict(model, x)
    next_index <- sample_mod(preds, diversity)
    nextword <- vocab[next_index]
    
    generated <- str_c(generated, nextword, sep = " ")
    sentence <- c(sentence[-1], nextword)
    
  }
  
  cat(generated)
  cat("\n\n")
  
}
model %>% save_model_hdf5("model2.h5", include_optimizer = TRUE)
load_model_hdf5("model2.h5")