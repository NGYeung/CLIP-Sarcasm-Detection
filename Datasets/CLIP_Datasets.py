# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:51:54 2024

@author: Yiyang Liu
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
import os, sys
import cv2
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from transformers import DistilBertTokenizer
import torch
from PIL import Image
import requests
from transformers import Dinov2Config, Dinov2Model
from transformers import ViTModel, ViTFeatureExtractor

class Meme_DataSet():
    
        
    def __init__(self, img_dir, text_file ='Datasets/TEXT_sentences.csv' ,img_model = 'DINOv2', text_model = 'DistilledBert'):
    
        

        self.text_path = text_file
        self.img_dir = img_dir
        self.text_tokenizer =None
        self.img_preprocess = None
        if img_model == 'DINOv2':
            self.img_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
        # Fill in the efficient net
        if text_model == 'DistilledBert':
            self.text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.images = []
        self.text = []

    def __len__(self):
        item_list = os.listdir(self.img_dir)
        return len(item_list)

    def __getitem__(self, idx):
        
        maxlength = 60 # REMEMBER TO ADJUST THIS!!!!!!!!
        text = pd.read_csv(self.text_path)
        self.text = list(text['text_corrected'])[idx]
      
        ct = len(os.listdir(self.img_dir)[idx])
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
     
        
        transform = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize((224, 224)),
                               transforms.ToTensor()
                                ])
        image_batch = transform(cv2.imread(img_path))
           
   
        if self.img_preprocess:
            image = self.img_preprocess(images = image_batch, return_tensors = 'pt')
 
            
        
        
        #CAPTION at entry 1

        text_batch = self.text_tokenizer.encode_plus(
            self.text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=maxlength,
            padding='max_length',
            truncation=True,  # Truncate to max_length
            return_attention_mask=True,  # Return attention masks
            return_tensors='pt'  # Return PyTorch tensors
            
            )
      
       
        return {'image': image['pixel_values'], 'input_ids': text_batch['input_ids'], 'attention': text_batch['attention_mask']}
    
    def getimages(self, idx):
        '''
        Used to return unprocessed image

        '''
        for i in idx:
            item = os.listdir(self.img_dir)[i]
            img_path = os.path.join(self.img_dir,item)
            image = cv2.imread(img_path)
            #image = cv2.resize(image,(224, 224,3))
            self.images.append(image)
            
        return self.images
    
    


        
