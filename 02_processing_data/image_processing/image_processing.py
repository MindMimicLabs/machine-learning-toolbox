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
