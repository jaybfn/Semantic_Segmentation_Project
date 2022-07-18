# Importing libraries
import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from train_sm import new_folder
sm.set_framework('tf.keras')
sm.framework()

print(new_folder)
#new_folder = 'NEW_EXP1'

def predict(image_dir_path):

    # Recursive calling for images
    image_dataset = []
    for directory_path in glob.glob(image_dir_path):
        for img_path in glob.glob(os.path.join(directory_path,"*.jpg")):
            img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)       
            img = cv2.resize(img, (HEIGHT, WIDTH))
            image_dataset.append(img)
        
    #Convert list to array for processing        
    image_dataset = np.array(image_dataset)
    #Images = image_dataset.shape[0]
    #print('Images:',Images)

    # data normalization
    scaler = MinMaxScaler()
    image_dataset = scaler.fit_transform(image_dataset.reshape(-1, image_dataset.shape[-1])).reshape(image_dataset.shape)   # fit and transform image using MinMaxScaler
    
    # preprocessing the input if a backbone is used else comment the line below if you want to use just UNET
    image_dataset = preprocessing_input(image_dataset)

    # Predicition
    test_pred_batch = model.predict(image_dataset)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)

    # Plotting results
    plt.figure(figsize=(18, 12))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(image_dataset[0])
    plt.subplot(232)
    plt.title('Prediction on Test Image')
    plt.imshow(test_pred_batch_argmax[0])
    figname = 'test.png'
    plt.savefig(path_predict+'/'+ figname)
    #plt.show()


if __name__=='__main__':

    # Model and its preprocessing inputs for images
    BACKBONE = 'resnet152'
    preprocessing_input = sm.get_preprocessing(BACKBONE)

    # path
    
    path = new_folder
    path_model = path+'/'+'model'
    path_predict = path+'/'+'predict_results'
    # Load Models
    model_name = 'model.h5'
    model = load_model(path_model+'/'+model_name, compile=False) #
    

    # Image parameters
    WIDTH = 256
    HEIGHT = 256
    n_classes = 9 

    # Image paths!, Where a single image to predict is stored!
    image_dir = '../../data/unet_img/Data_TF_Scalerapid_Split/test_image/Original/'

    predict(image_dir)