"""
Created on Thu Aug 5

@author: Yiyang Liu


"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
from PIL import Image
import os, sys
from wordcloud import WordCloud, STOPWORDS
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
#import multiprocessing
#from gensim.models.doc2vec import Doc2Vec
#from gensim.models.phrases import Phrases, Phraser
#import gensim.downloader as api
#from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel





"""
Image-Text-Contrastive loss.
Reference: https://arxiv.org/pdf/2103.00020v1 the CLIP Paper
Reference for the "target": https://arxiv.org/pdf/2103.00020v1
"""

class I_T_ContrastiveLoss_wl(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.scale = 1.0/temperature


    def logits(self, image_features, text_features):
        V_image = self.scale * image_features @ text_features.T
        V_text = self.scale * text_features @ image_features.T
        return V_image, V_text

    def forward(self, image_features, text_features, labels,  visualize = 0):
        
        device = image_features.device
        V_image, V_text = self.logits(image_features, text_features)
        dim = image_features.size(0) #need to check the dimension here 

        
        targets = F.softmax(self.scale * (image_features @ image_features.T + text_features @ text_features.T)/2, dim = -1)

        
        #----------------------------
        
        if visualize:
            sns.heatmap(V_text.detach().numpy(), annot=False, cmap='viridis')


            plt.title('TEST - logits')

            plt.show()
            
            
            sns.heatmap(targets.detach().numpy(), annot=False, cmap='viridis')


            plt.title('TEST - targets')

            plt.show()
        
        
        #----------------------------------
        total_loss = (F.cross_entropy(V_text, targets,reduction='none') + F.cross_entropy(V_image, targets.T ,reduction='none')  )/2
       

        return total_loss





"""
Image-Text-Contrastive loss.
Reference: https://arxiv.org/pdf/2103.00020v1 the CLIP Paper
Reference for the "target": https://arxiv.org/pdf/2103.00020v1
"""

class I_T_ContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.scale = 1.0/temperature


    def logits(self, image_features, text_features):
        V_image = self.scale * image_features @ text_features.T
        V_text = self.scale * text_features @ image_features.T
        return V_image, V_text

    def forward(self, image_features, text_features, visualize = 0):
        
        device = image_features.device
        V_image, V_text = self.logits(image_features, text_features)
        dim = image_features.size(0) #need to check the dimension here 

        
        targets = F.softmax(self.scale * (image_features @ image_features.T + text_features @ text_features.T)/2, dim = -1)
        
  
		
        
        
        #----------------------------
        
        if visualize:
            sns.heatmap(V_text.detach().numpy(), annot=False, cmap='viridis')


            plt.title('TEST - logits')

            plt.show()
            
            
            sns.heatmap(targets.detach().numpy(), annot=False, cmap='viridis')


            plt.title('TEST - targets')

            plt.show()
        
        
        #----------------------------------
        total_loss = (F.cross_entropy(V_text, targets,reduction='none') + F.cross_entropy(V_image, targets.T ,reduction='none')  )/2
        
        
     
        

        return total_loss.mean()







"""
CLIP with 1) Doc2Vec / BERT - DINOv2 / TBD
"""  
class clip_for_meme(nn.Module):
    def __init__(self, text_encoder='BERT', image_encoder='DINOv2', embedding_size=768 ,projection_size=1024):
        
        '''
        text_encoder = 'Doc2Vec' or 'BERT'
        image_encoder = 'DINOv2'
        '''
        
        super(clip_for_meme, self).__init__()
        
        #since we don't know the oterh  model yet I will just use dinov2 by default
        if image_encoder == 'DINOv2':
            self.image_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.img_projection = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(embedding_size, projection_size),
                nn.GELU(),
                nn.Linear(projection_size, projection_size),
                nn.LayerNorm(projection_size)
                )
            
        if text_encoder == 'BERT':
            self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.txt_projection = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(embedding_size, projection_size),
                nn.GELU(),
                nn.Linear(projection_size, projection_size),
                nn.LayerNorm(projection_size)
                )
        '''
        if text_encoder == 'Doc2Vec':
            # not properly debugged yet. Also no good pre-trained doc2vec to use :/
            self.text_model = Doc2Vec(vector_size=vector_len,  
                window=2,         
                min_count=1,      
                workers=4,       
                epochs=6)
            self.txt_projection = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(n_size, vector_len),
                )
            #need an extra method to train the Doc2Vec because, again, no good pre-trained model to use
            '''
        

    
    def forward(self, images, input_ids, attention_mask):
        image_embed = self.image_model(images)
        emb_len = image_embed.size(1)
        image_embed = self.img_projection(image_embed)

        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_out['last_hidden_state'][:,0,:] # [CLS] tokens at index 
        text_embed = self.txt_projection(text_embed)

        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1)
    
    
    
    
    
