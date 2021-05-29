# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:20:47 2021

@author: Raktim
"""
#Data Load for Segmentation
#PATH should contain TWO folder
#ONE is IMAGE folder and another one MASK folder
#This same structure has to be for both train and test image

#train_path
########### Image
########### Mask

#test_path
########## Image
########## Mask

import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.utils import to_categorical


class Data():
    
    def __init__(self): 
        pass
    
    def load_segmentation_data(self, img_path, file_types, img_h, img_w, channels=3):
        
        image_count = sum(len(files) for _, _, files in os.walk(img_path))
        image_count= int(image_count/2) #half because 50% image and 50% mask
        #And we want either image or mask image number 
        ids = next(os.walk(img_path))[1]
        image = np.zeros((image_count, img_h, img_w, channels), dtype=np.float32)
        mask = np.zeros((image_count, img_h, img_w, 1), dtype=np.uint8)
        
        i=0
        j=0
        
        print('Importing + Resizing Images  Masks')
        for id_ in tqdm(ids):
            path = img_path + id_
            
            for image_file in next(os.walk(path + '/image/'))[2]:
                if (image_file.split('.')[1] == file_types):
                    image_ = imread(path + '/image/' + image_file)
                    image_=resize(image_, (img_h, img_w), anti_aliasing=True)
                    image[i] = img_as_float(image_)
                    i=i+1
                    
            for mask_file in next(os.walk(path + '/mask/'))[2]:
                if (mask_file.split('.')[1] == file_types):
                    mask_ = imread(path + '/mask/' + mask_file)
                    mask_= resize(mask_, (img_h, img_w), anti_aliasing=True)
                    grayscale = np.expand_dims(rgb2gray(mask_),axis=-1)
                    mask[j] = grayscale
                    j=j+1
                    
        return image, mask
    
    def load_classification_data(self, img_path, file_types, img_h, img_w):

        dataset = []  
        label = []  
        
        ids = next(os.walk(img_path))[1]
        print('Importing Images and Assigning Labels')
        for n, id_ in tqdm(enumerate(ids)):
            path = img_path + id_ +'/'
            for image_file in next(os.walk(path))[2]:
                if (image_file.split('.')[1] == file_types):
                    image_ = imread((path + '/'+image_file), as_gray=False)
                    image_ = resize(image_, (img_h, img_w), anti_aliasing=True)
                    dataset.append(img_as_float(image_))
                    label.append(n)
        data = np.array(dataset)
        label = to_categorical(np.array(label))
        
        return data, label


        
        