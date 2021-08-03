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