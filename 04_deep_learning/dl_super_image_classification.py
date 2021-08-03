import cv2
import os
import sys
from tqdm import tqdm  
import numpy as np
import csv
#import argparse
import random
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

IMG_SIZE = 80

def create_list(DIR, IMG_SIZE):
    X_training_data = []
    target = []
    import cv2
    import numpy as np
    from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
    for inp in [0,1]:
        TRAIN_DIR= DIR+"/"+str(inp)+"/"
        for img in tqdm(os.listdir(TRAIN_DIR)):
            path = os.path.join(TRAIN_DIR, img)
            #img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img_data = cv2.imread(path, cv2.IMREAD_COLOR)
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE),3)
            X_training_data.append(np.array(img_data))
            target.append(inp)

    y_training_data = to_categorical(target, num_classes = 2)
    #y_training_data = target

    img = None
    img_data = None
    path = None

    y_train = None
    X_train = None
    x_train = None
    trainX, testX, trainY, testY = None, None, None, None
    H, model= None, None

    X_train = np.array(X_training_data)
    x_train = X_train/255.0
    y_train = np.array(y_training_data)
    X_train = x_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 3)
    x_train = None
    
    return(X_train, y_train)

trainX, trainY = create_list("/data/images/train/", IMG_SIZE)
testX, testY = create_list("/data/images/valid/", IMG_SIZE)
print('There are', trainX.shape[0], 'training data and', testX.shape[0], 'testing data')

#### More complesx model ########### #################
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="sigmoid"):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))
        
        
        # first (and only) set of FC => RELU layers

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        
        # use a *softmax* activation for single-label classification
        # and *sigmoid* activation for multi-label classification
        model.add(Dense(classes))
        model.add(Activation(finalAct))
 
        # return the constructed network architecture
        return model
		# Done with model
import matplotlib
import sklearn
 
##import matplotlib.pyplot as plt
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 180
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (IMG_SIZE, IMG_SIZE, 3)

mlb = MultiLabelBinarizer()
aug = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,height_shift_range=0.2, horizontal_flip=True)
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    ##    plt.show()
    plt.savefig('images/plot1.png', format='png')
import time
start_time = time.time()

# initialize the optimizer
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience= 40)
logger = callbacks.CSVLogger('data/log.csv', separator=',', append=False)
callback=[early_stop, logger]
###########################################
import numpy as np
speciv = []
sensitiv = []
accu = []
observ = []
accuy =[]
X= np.concatenate((trainX, testX))
y= np.concatenate((trainY, testY))
X_train = None
y_train = None
n= X.shape[0]
from sklearn.model_selection import KFold
#kf = KFold(n, n_splits=10, shuffle=True)
kf = KFold(n_splits=10, shuffle=True)
issues = ['1', '0']
accur = [] 

#for iteration, data in enumerate(kf, start=1):
for train_index, test_index in kf.split(X):

	trainX = X[train_index]
	testX = X[test_index]
	trainY = y[train_index] 
	testY = y[test_index]

	model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=2, finalAct="sigmoid")
	# Compile model
	model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])	# Fit the model
	(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(valX, valY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, callbacks= callback,verbose=0)
	scores = model.evaluate(testX , testY , verbose=1)
	print("#########################################")
	print("Accuracy= ", scores[1])
	accu.append(scores[1])
	print(accu)

##plt.show()
rsd = (time.time() - start_time)/3600.0

print("--- %s hours ---" %rsd)

!mkdir -p saved_model
model.save('saved_model/model')
model.save()

# summarize feature map size for each conv layer
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
# load the model
model = model
# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
	
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

# cannot easily visualize filters lower down
from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
# load the model
model = model
# retrieve weights from the second hidden layer
filters, biases = model.layers[0].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='jet')
		ix += 1
# show the figure
pyplot.show()

from keras.models import Model
pred = model.predict(testX) 
y_classes = pred.argmax(axis=-1) 
pred2 = model.predict_classes(testX)

print(pred[1])
print(y_classes)
print(pred2)

from sklearn import metrics
import numpy as np
rounded_labels=np.argmax(testY, axis=1)
cm = metrics.confusion_matrix(rounded_labels, y_classes)
print(cm)

import matplotlib.pyplot as plt
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

print(metrics.classification_report(rounded_labels, y_classes))

print(sklearn.metrics.roc_auc_score(rounded_labels, y_classes))

from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
# load an image from file
image = load_img('data/images/valid/1/-----.png', target_size=(80, 80))
plt.imshow(image)
plt.title('ORIGINAL IMAGE')

from keras.applications.vgg16 import preprocess_input
from keras.models import Model
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
img = cv2.imread('data/images/valid/1/-----.png')
img = cv2.resize(img,(80,80))
img = np.expand_dims(img, axis=0)
print(img.shape)
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 4
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='jet')
		ix += 1
# show the figure
pyplot.show()
