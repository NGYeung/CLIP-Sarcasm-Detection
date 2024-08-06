# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:50:40 2024

@author: Yiyang Liu

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA


    
class Visualizer():
    """
    This class is used to visualize the word-embedding.
    """
    
    def __init__(self, sent, embedding, idx):
        """
        Parameters
        ----------
        key [list]
            DESCRIPTION.
        embedding [list]
            DESCRIPTION.
        idx [list]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.words = sent
        self.vectors = embedding
        self.idx = idx
    
        
    def show(self):
        """show the image! """

        TD = PCA(n_components = 2)
        x1,y1 = TD.fit_transform(self.vectors).T
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x1[self.idx],y1[self.idx], c='k',s = 3)

        for i in self.idx:
            ax.annotate(
                self.words[i],
                (x1[i], y1[i]),
                xytext=(2, 2),
                textcoords='offset points',fontsize = 6
                )

        plt.title('word embeddings')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()