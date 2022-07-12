# calling libraries
import os
import random
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

sm.set_framework('tf.keras')

sm.framework()

new_folder = '07July2022' 


# create directory to save the data
def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

                                                                     
# preprocessing the input images
def data_preprocessing(img, mask, num_class):
    
    #Scale images  
   
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)   # fit and transform image using MinMaxScaler
    
    # preprocessing the input if a backbone is used else comment the line below if you want to use just UNET
    img = preprocessing_input(img)
    
    # label encoding for the mask image
    labelencoder = LabelEncoder()                                                   # initializing Labelencoder
    number_of_images, height, width, channles= mask.shape                           # shape of the mask image
    mask_reshape = mask.reshape(-1,1)                                               # reshaping the mask image numpy array
    encoded_mask = labelencoder.fit_transform(mask_reshape.ravel())                 # fit and transform mask image using label encoder
    original_encoded_mask = encoded_mask.reshape(number_of_images, height, width )  # reshaping the image numpy array
    mask = np.expand_dims(original_encoded_mask, axis = 3)                          # expanding dimension (requirement by the model)
                                                                    
    #Convert mask to one-hot encoding
    mask = to_categorical(mask, num_class)
                                                                                    # into to categorical pixel values
    return (img,mask)


# defining the data loader
def TFDataLoader(train_img_path, train_mask_path, num_class):
    
    # augmention parameters for images
    img_data_gen_args = dict(
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                      )

    # initializing ImageDataGenerator for both images and masks
    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)
    
    # images will be loaded directly from the local drive (less load on the memory)
    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        target_size=(256, 256),                                                     # for PSP net ,shape should be divisible by 48
        class_mode = None,
        color_mode = "rgb",
        batch_size = batch_size,
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size=(256, 256),
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)
        
    # zip both images and mask 
    data_generator = zip(image_generator, mask_generator)
    
    for (img, mask) in data_generator:
        img, mask = data_preprocessing(img, mask, num_class)
        #mask = mask[:,:,:,1:]                                                      # to remove background!
        yield (img, mask)

def metricplot(df, xlab, ylab_1,ylab_2, path):
    
    """
    This function plots metric curves and saves it
    to respective folder
    inputs: df : pandas dataframe 
            xlab: x-axis
            ylab_1 : yaxis_1
            ylab_2 : yaxis_2
            path: full path for saving the plot
            """
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = df[xlab], y = df[ylab_1])
    sns.lineplot(x = df[xlab], y = df[ylab_2])
    plt.xlabel('Epochs',fontsize = 12)
    plt.ylabel(ylab_1,fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend([ylab_1,ylab_2], prop={"size":12})
    plt.savefig(path+'/'+ ylab_1)
    #plt.show()
    
if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)

    # data normalization
    scaler = MinMaxScaler()


    BACKBONE= 'resnet152'                                                            # Backbone 
    preprocessing_input = sm.get_preprocessing(BACKBONE)
    seed=24                                                                          # setting seed 
    batch_size= 4                                                                    # batch_size for training
    n_classes=9                                                                      # number of class/ labels in the dataset
    epochs = 300                                                                     # epochs to run                                                          
    learning_rate = 1e-5                                                             # learning_rate
    

    # define a new folder name to save data
                                                                
    path = new_folder                                                                # to save all complete experiment 
    path_metric = path+'/'+'metricplots'                                             # to save all the trained metricplots and plot data
    path_model = path+'/'+'model'                                                    # model.h5 file will be save here
    path_test_metric = path+'/'+'test_results'                                       # save all the test results
    path_predict = path+'/'+'predict_results'
    create_dir(path)                                                                 # creating directory to save new experiment
    create_dir(path_metric)                                                          # creating directory to save all the metric from the trained data
    create_dir(path_model)
    create_dir(path_test_metric)
    create_dir(path_predict)
    model_name = 'model.h5'

    # path for both train, val datasets and test dataset for scale.rapid images

    train_img_path = "DataSet/train_image/"                    
    train_mask_path = "DataSet/train_mask/"
    train_img_gen = TFDataLoader(train_img_path, train_mask_path, num_class=n_classes)     # calling TFDataLoader for training datasets  

    val_img_path = "DataSet/val_image/"
    val_mask_path = "DataSet/val_mask/"
    val_img_gen = TFDataLoader(val_img_path, val_mask_path, num_class=n_classes)  

    # data iterator
    x_train, y_train = train_img_gen.__next__()                                       # Training dataset
    x_val, y_val = val_img_gen.__next__()                                             # Validation dataset

    # inputs for the UNET model
    IMG_HEIGHT = x_train.shape[1]
    IMG_WIDTH  = x_train.shape[2]
    IMG_CHANNELS = x_train.shape[3]
    #print(IMG_CHANNELS)

    # defining the loss function:
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1*focal_loss)

    metrics = [sm.metrics.IOUScore(threshold = 0.5), sm.metrics.FScore(threshold = 0.5),'accuracy', 'Recall', 'Precision']

    keras.backend.clear_session()

    # training image path
    train_img_path_len = "DataSet/train_image/Original"
    img_list_len = len(os.listdir(train_img_path_len))
    #print(img_list_len)

    # input for the model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # model intializing (using UNET from Segmentation model lib)    
    model = sm.Unet(BACKBONE, 
                    encoder_weights = 'imagenet', 
                    input_shape = input_shape, 
                    classes = n_classes ,
                    activation = 'softmax')
    
    # loading weights:
    #model.load_weights(path_model+'/'+model_name) 

    # model compiling:
    model.compile(optimizer = Adam(learning_rate = learning_rate),
                    loss = total_loss, 
                    metrics = metrics)
    #model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(path_model+'/'+model_name),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100),
        tf.keras.callbacks.CSVLogger(path_metric+'/'+'data.csv'),
        tf.keras.callbacks.TensorBoard(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125, restore_best_weights=False)]
    
    # model fitting
    steps_per_epoch = img_list_len //batch_size
    history = model.fit(train_img_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs = epochs,
                        validation_data=val_img_gen, 
                        validation_steps=steps_per_epoch,
                        verbose = 1, callbacks = callbacks )

    #Model Evaluation
    print("____________________________________________________________")
    print("____________________________________________________________")
    print("____________________________________________________________")
    train_IoU = model.evaluate(train_img_gen,
                                    batch_size = batch_size,
                                    steps = steps_per_epoch)
    print("Train IoU is = ", (train_IoU[1] * 100.0), "%")

    val_IoU = model.evaluate(val_img_gen,
                                    batch_size = batch_size,
                                    steps = steps_per_epoch)
    print("Val IoU is = ", (val_IoU[1] * 100.0), "%")

    print("____________________________________________________________")
    print("____________________________________________________________")
    print("____________________________________________________________")

    model.save(path_model+'/'+model_name)
    df = pd.read_csv(path_metric+'/'+ 'data.csv')

    # plot metric_curve
    metricplot(df,'epoch', 'accuracy', 'val_accuracy', path_metric)
    metricplot(df,'epoch', 'iou_score', 'val_iou_score', path_metric)
    metricplot(df,'epoch', 'loss', 'val_loss', path_metric)
    metricplot(df,'epoch', 'precision', 'val_precision', path_metric)
    metricplot(df,'epoch', 'recall', 'val_recall', path_metric)

