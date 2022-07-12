# Importing libraries

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU
import segmentation_models as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from train_sm import new_folder
sm.set_framework('tf.keras')
sm.framework()

print(new_folder)
#new_folder = 'NEW_EXP1'

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

# calculating IoU for individual labels
def IoU_classes(n_classes,weight_values, classes):
    """ Calculate IoU for each class or label"""
    # initializing a dict to store all the IoU values for each label or class
    IoU_individual_classes = {}
    for i , j, label in zip(np.arange(n_classes), np.arange(n_classes), classes):
        IoU_individual_classes[label] = weight_values[i,j]/(np.sum(weight_values[:,i]) + np.sum(weight_values[j]) - weight_values[i,j])
    IoU_all_classes = pd.DataFrame([IoU_individual_classes])
    return IoU_all_classes

# defining function for multilabel confusion matrix
def MultiClassConfusionMatrix(corr_data, classes):
    ground_truth = corr_data.sum(axis = 1)                  # summing all the ground truth (sum all the rows)                             
    corr_data_norm = corr_data/ground_truth.reshape(-1,1)   # normalizing 
    corr_data_norm_df = pd.DataFrame(corr_data_norm,        # building a dataframe
                                    index = classes, 
                                    columns= classes)
    plt.figure(figsize=(10, 10))                            # setting the size of the figure
    corr_mat = sns.heatmap(corr_data_norm_df,               # plotting 
                        annot=True, 
                        fmt=".3f",
                        cmap='BuPu')
    figname = 'ConfussionMatrix_TestSet.png'
    plt.savefig(path_test_metric+'/'+figname)
    return corr_mat
    

if __name__=='__main__':

    np.random.seed(42)
    tf.random.set_seed(42)  

    
    # create_dir(path_test_metric)
    # data normalization
    scaler = MinMaxScaler()

    # Model and its preprocessing inputs for images
    batch_size= 4
    seed=24  
    BACKBONE = 'resnet50'
    preprocessing_input = sm.get_preprocessing(BACKBONE)

    # path
    path = new_folder
    path_model = path+'/'+'model'
    path_test_metric = path+'/'+'test_results' 
    # Load Models
    model_name = 'model.h5'
    model = load_model(path_model+'/'+model_name, compile=False)

    test_img_path = "DataSet/test_image/"
    test_mask_path = "DataSet/test_mask/"
    test_img_gen = TFDataLoader(test_img_path, test_mask_path, num_class=9)

    test_image_batch, test_mask_batch = test_img_gen.__next__()

    #Convert categorical to integer for visualization and IoU calculation
    test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3) 
    test_pred_batch = model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

    # calculating MeanIoU (intersection over Union)
    n_classes = 9
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    # calculating classwise IoU for the test dataset
    weight_values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    #classes = ['Background','Wings','Proboscis','Head','Antennae','Palps','Abdomen','Legs','Thorax'] # for cvat
    classes = ['Background','Palps','Abdomen','Thorax','Antennae','Wing','Head','Proboscis','Legs'] # for scale.rapid
    weight_val_df = pd.DataFrame(weight_values, index = classes, columns= classes)   

    IoU_classes = IoU_classes(n_classes,weight_values,classes)
    filename_csv = 'IoU_ClassWise_TestSet.csv'
    IoU_classes.to_csv(path_test_metric+'/'+ filename_csv)
    print(IoU_classes)

    # Confusion Matrix for the test dataset
    MultiClassConfusionMatrix(weight_values,classes)

    #View a few images, masks and corresponding predictions. 
    img_list = random.sample(range(0, test_image_batch.shape[0]), batch_size)

    for img_num in img_list:
        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_image_batch[img_num])
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(test_mask_batch_argmax[img_num])
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(test_pred_batch_argmax[img_num])
        figname = f'{img_num}.png'
        plt.savefig(path_test_metric+'/'+figname)
        #plt.show()