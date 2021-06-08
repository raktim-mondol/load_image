#!/usr/bin/env python3  
"""
Created on Thu May 27 20:20:47 2021

@author: Raktim
"""
#Data Load for Segmentation
#PATH should contain TWO folder
#ONE is IMAGE folder and another one MASK folder
#This same structure has to be for both train and test image

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.utils import to_categorical

# use forward slash (/) when provide path
class Data():
    
    def __init__(self): 
        pass
    
    def load_segmentation_data(self, img_path, file_types, img_h, img_w, channels=3):
        self.img_path = img_path
        ids = next(os.walk(img_path))[1]
        image=[]
        mask=[]
        
        print('Importing + Resizing Images  Masks')
        for id_ in tqdm(ids):
            path = img_path + id_
            
            for image_file in next(os.walk(path + '/image/'))[2]:
                if (image_file.split('.')[1] == file_types):
                    image_ = cv2.imread((path + '/image/' + image_file),cv2.IMREAD_COLOR)
                    if (img_h>image_.shape[0]):
                        interpolation=cv2.INTER_CUBIC
                        #upscale image
                    elif (img_h==image_.shape[0]):
                        interpolation=cv2.INTER_LINEAR
                        #no change in size
                    else:
                        interpolation=cv2.INTER_AREA
                        #downscale image
                        #cv2.INTER_NEAREST   also good 
                    image_ = cv2.cvtColor(image_,cv2.COLOR_BGR2RGB)
                    image_= cv2.resize(image_, (img_h, img_w), interpolation = interpolation)
                    #image[i] = img_as_float(image_)
                    image.append(image_)

                    
            for mask_file in next(os.walk(path + '/mask/'))[2]:
                if (mask_file.split('.')[1] == file_types):
                    mask_ = cv2.imread((path + '/mask/' + mask_file), cv2.IMREAD_GRAYSCALE)
                    if (img_h>mask_.shape[0]):
                        interpolation=cv2.INTER_CUBIC
                        #upscale image
                    elif (img_h==mask_.shape[0]):
                        interpolation=cv2.INTER_LINEAR
                        #no change in image
                    else:
                        interpolation=cv2.INTER_AREA
                        #downscale image
                        #cv2.INTER_NEAREST   also good 
                    mask_ = cv2.resize(mask_, (img_h, img_w), interpolation = interpolation)
                    mask_ = np.expand_dims(mask_, axis=-1)
                    mask.append(mask_)

                    
        image = (np.array(image, dtype=np.uint8)/255.).astype(np.float32)
        mask_ = np.array(mask, dtype=np.uint8)/255.
        mask = (mask_ > 0.2).astype(np.uint8)
        return image, mask
    
    def load_classification_data(self, img_path, file_types, img_h, img_w, to_cat=True):

        dataset = []  
        label = []  
        self.img_path = img_path
        self.label_name = []
        self.label_title = []
        
        #img_path='C:/D_DRIVE/UNSW/Experiment/Transfer_Learning/Dataset/Cats_and_Dogs/train/'
        
        ids = next(os.walk(img_path))[1]
        
        print('Importing Images and Assigning Labels')
        for n, id_ in tqdm(enumerate(ids)):
            path = img_path + id_
            self.label_name.append(id_)
            self.label_title.append(n)
            for image_file in next(os.walk(path+'/'))[2]:
                
                if (image_file.split('.')[1] == file_types):
                    #ABOUT image_file.split('.')[1]
                    #if there is only one .(dot) (which is before file_types) in the image then put [1]
                    #if there is One(1) extra .(dot) in the image (excluding dot(.) before file_types) then put [2]
                    image_ = cv2.imread((path +'/'+image_file), cv2.IMREAD_COLOR)
                    if (img_h > image_.shape[0]):
                        interpolation = cv2.INTER_CUBIC
                        #upscale image
                    elif (img_h==image_.shape[0]):
                        interpolation = cv2.INTER_LINEAR
                        #no change in size
                    else:
                        interpolation = cv2.INTER_AREA
                        #downscale image
                        #cv2.INTER_NEAREST   also good 
                    image_ = cv2.cvtColor(image_,cv2.COLOR_BGR2RGB)
                    image_ = cv2.resize(image_, (img_h, img_w), interpolation = interpolation)
                    dataset.append(image_)
                    label.append(n)
        
        data = (np.array(dataset, dtype=np.uint8)/255.).astype(np.float32)
        
        if (to_cat == True):
            label=to_categorical(np.array(label, dtype=np.uint8))
        else:
            label = np.array(label, dtype=np.uint8)
            
        return data, label
    
    
    def label_check(self):
        self.label_name
        self.label_title
        #print the label name with correspoing label
        for n, id_ in enumerate(self.label_name):
            print(str(self.label_title[n]) + ' ------>>>> ' + str(self.label_name[n]+"\n"))
            
    
    def visualize(self,image, mask=None):
        #to count the image 
        image_count = sum(len(files) for _, _, files in os.walk(self.img_path))
        
        #image with mask for segmentation 
        if mask is not None:
            image_count_= int(image_count/2)
            #divided by half because folder contain both images and masks
            #but we only need no. of images in any folder
            #so divided by two
            
            #this image_count(value) set as a limit for random number generator
            #which further used for random data visualization
            image_ = random.randint(0, image_count_-1)
            imshow(image[image_])
            plt.title('Actual Image')
            plt.show()
            imshow(mask[image_],cmap='gray')
            plt.title('Corresponding Mask')
            plt.show()
        #only image for classification
        else:
            image_ = random.randint(0, image_count-1)
            imshow(image[image_])
            plt.show()
            
    
