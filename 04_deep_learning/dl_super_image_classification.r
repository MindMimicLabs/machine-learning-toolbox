# - Packages...
library(keras)
library(tensorflow)
library(reticulate)
library(tfruns)
library(viridis) 

# - [1] - Model 1===================================================
start <- Sys.time()
image_list <- c("0", "1") 

# number of output classes (i.e. Classes (0 & 1))
output_n <- length(image_list)


# image size to scale down to (original images are 50 x 50 px)
img_width <- 50
img_height <- 50
target_size <- c(img_width, img_height)

# RGB = 3 channels
channels <- 3

# path to image folders
path.train = "data/images/train"
train_image_files_path <- file.path(path.train)
path.valid = "data/images/valid"
valid_image_files_path <- file.path(path.valid)

# optional data augmentation
train_data_gen = image_data_generator(
  rescale = 1/255,
  featurewise_center = T,
  featurewise_std_normalization = T,
  samplewise_std_normalization = T,
  rotation_range = 2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=TRUE,
  vertical_flip = TRUE,
  zca_whitening = TRUE,
  validation_split = 0.2
  #augment = TRUE
)

# Validation data shouldn't be augmented! But it should also be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255,
  validation_split = 0.2
)  

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    generator =train_data_gen,
                                                    target_size = target_size,
                                                    color_mode = "rgb",
                                                    class_mode = 'categorical',
                                                    shuffle=TRUE,
                                                    classes = image_list,
                                                    subset = 'training',
                                                    #interpolation = "bilinear",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    generator = valid_data_gen,
                                                    target_size = target_size,
                                                    color_mode = "rgb",
                                                    class_mode = 'categorical',
                                                    shuffle=TRUE,
                                                    classes = image_list,
                                                    subset = 'validation',
                                                    #interpolation = "bilinear",
                                                    seed = 42)


test_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                   generator = valid_data_gen,
                                                   target_size = target_size,
                                                   color_mode = "rgb",
                                                   class_mode = 'categorical',
                                                   shuffle=TRUE,
                                                   classes = image_list,
                                                   subset = 'validation',
                                                   #interpolation = "bilinear",
                                                   seed = 42)

### model definition

# number of training samples
train_samples <- train_image_array_gen$n
train_samples

# number of validation samples
valid_samples <- valid_image_array_gen$n
valid_samples

# define batch size and number of epochs
batch_size <- 2
epochs <- 10
filters <- 128

# initialise model
model <- keras_model_sequential()
K <- backend()
model %>% 
  layer_conv_2d(filter = filters,kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last",
                input_shape = c(img_width, img_height, channels)) %>%
  
  # add layers
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters/2, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # add layers
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters/2, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  
  # add layers
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # add layers
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  # add layers
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same',
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_conv_2d(filter = filters, kernel_size=c(3,3), padding = 'same', 
                dilation_rate = c(1L, 1L), data_format="channels_last") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  layer_dense(filters) %>%
  layer_dropout(0.50) %>%
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>% 
  layer_activation('sigmoid') #sigmoid


# compile
model %>% compile(
  loss = 'binary_crossentropy', #binary
  optimizer = optimizer_nadam(lr = 0.002), 
  metrics = 'accuracy'
)

summary(model)


# fit
for (i in 1:10){
  hist <- model %>% fit_generator(
    # training data
    train_image_array_gen,
    
    # epochs
    steps_per_epoch = as.integer(train_samples/batch_size), 
    epochs = epochs,
    
    # validation data
    validation_data = valid_image_array_gen,
    validation_steps = as.integer(valid_samples/batch_size),
    verbose = 1,
    max_queue_size=10,
    workers=5,
    callbacks = list(callback_early_stopping(patience = 10, monitor = 'val_loss', restore_best_weights = TRUE),
                     callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.01, 
                                                   patience=9, verbose=1, mode='auto', 
                                                   min_delta=1e-6, cooldown=0, min_lr=1e-10))
  )
  
  # number of validation samples
  test_samples <- test_image_array_gen$n
  # evaluate accuracy of model1 using testing data
  evaluation = model %>% evaluate_generator(test_image_array_gen, steps = test_samples/batch_size)
  evaluation
  
  model %>% reset_states()
  cat("Epoch: ", i)
}

plot(hist)


df_out <- hist$metrics %>% 
  {data.frame(acc = .$acc[epochs], val_acc = .$val_acc[epochs], 
              elapsed_time = as.integer(Sys.time()) - as.integer(start))}

df_out



# predict the probability of class

filenames = test_image_array_gen$filenames
samples = length(filenames)
prob = model %>% predict_generator(test_image_array_gen, steps = samples, verbose = 1)
prob



# - [2] - Save the Model for Further Training===================================
model %>% save_model_hdf5("model3.h5")
load_model_hdf5("model3.h5")
model %>% summary()