import numpy as np
from skimage import io
import random
import os
from scipy.ndimage import rotate
from tqdm import tqdm

number_of_images = "../../data/unet_img/Data_TF_Scalerapid_Split/train_image/Original/"
img_list_len = len(os.listdir(number_of_images))

images_to_generate = img_list_len
seed_for_random = 42




#Define functions for each operation
# Make sure the order of the spline interpolation is 0, default is 3. 
#With interpolation, the pixel values get messed up.
def rotation(image, seed):
    random.seed(seed)
    angle= random.randint(-180,180)
    r_img = rotate(image, angle, mode='constant', cval = 0, reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img= np.fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img= np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img



transformations = {'rotate': rotation,
                'horizontal flip': h_flip, 
                'vertical flip': v_flip,
                'vertical shift': v_transl,
                'horizontal shift': h_transl
                 }                #use dictionary to store names of functions 

images_path="../../data/unet_img/Data_TF_397/val_image/JPEGImages/" #path to original images
masks_path = "../../data/unet_img/Data_TF_397/val_mask/SegmentationClass/"
img_augmented_path="../../data/unet_img/JPEGImages_aug/" # path to store aumented images
msk_augmented_path="../../data/unet_img/SegmentationClass_aug/" # path to store aumented images
images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path): 
    #print(im) # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))
print(len(images))
for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))

for i, im, msk in tqdm(zip(range(images_to_generate), os.listdir(images_path), os.listdir(masks_path))):
    img_name = im.split('/')[-1].split('.')
    msk_name = msk.split('/')[-1].split('.')
    img = img_name[0]
    msk = msk_name[0]
    
    image = images[i]
    mask = masks[i]
    original_image = io.imread(image)
    original_mask = io.imread(mask)
    transformed_image = None
    transformed_mask = None

    for key in transformations:
        seed = random.randint(1,100)  #Generate seed to supply transformation functions. 
        transformed_image = transformations[key](original_image, seed)
        transformed_mask = transformations[key](original_mask, seed)
        #n = n + 1
        
        new_image_path= img_augmented_path + f'{img}_{key}.png' 
        new_mask_path = msk_augmented_path+ f'{msk}_{key}.png'    #Do not save as JPG
        io.imsave(new_image_path, transformed_image)
        io.imsave(new_mask_path, transformed_mask)
        i =i+1