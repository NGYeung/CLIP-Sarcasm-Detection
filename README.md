# A meme ranking system using constractive learning

This project aims to the build a ranking system for <b> MEMES </b>

<img src="https://static.wixstatic.com/media/bb1bd6_5798c09022ba43249a38bfea9be1db34~mv2.png/v1/fill/w_980,h_560,al_c,q_90,usm_0.66_1.00_0.01,enc_auto/bb1bd6_5798c09022ba43249a38bfea9be1db34~mv2.png" width="400">


# Methodology
*INPUT: The caption of a <b> MEMES </b>*

*OUTPUT: The top 10 most relevant memes in the dataset ranked by a similarity score that weighs both text-text similarity and text-image correlation.*

*MODEL: A variation of <a href="https://openai.com/index/clip/">CLIP by OPEN AI</a> with Gensim Doc2Vec/Hugging Face DistilledBERT and Facebook DINOv2/Google AI EfficientNet as the text and image encoders, respectively.*


Here is a flow chart drawn by an awesome artist. (ME.)


<img src = 'https://github.com/NGYeung/summer-2024-meme-ranking/blob/711622da3760d8c1f3522b0912d323ce95fb32a9/readme-images/flowchart.jpg'>

<br>


# Dataset

We used the <a href ='https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k'> MEMOTION DATASET 7K </a> as our training and testing dataset. 

Selected features: Images and Captions.

Training set size: 7000

Testing set size: 2000

# Structure of the repository

*The implementation of clip model is in the "custom_models" folder. *

*The trainer module "CLIP_trainer.py" and training note of the model is in the root folder.*

*The Datasets folder includes a sample of images and texts and the Dataset class in CLIP_Datasets.py*
