"""
Created on Thu Aug 7

@author: Yiyang Liu

"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_model.clip import clip_for_meme, I_T_ContrastiveLoss
from Datasets.CLIP_Datasets import Meme_DataSet # Custom Dataset
import os, sys



def one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    
    model.train()

    ###Iterating over data loader
    for i, (images, input_ids, attention_mask, tokentype_id) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        img_features, txt_features = model(images, input_ids, attention_mask)
        #Calculating Loss
        b_loss = loss_fn(img_features, txt_features)
        epoch_loss.append(b_loss.item())      
        #Backward
        b_loss.backward() #compute gradients
        optimizer.step()

        # calculate acc per minibatch
        #_,text_logits = loss_fn.logits(img_features, txt_features)
        # The following quoted because we don't need labels.
        #labels = loss_fn.get_ground_truth(img_features.device, txt_features.shape[0])
        #sum_correct_pred += (torch.argmax(logits,dim=-1) == labels).sum().item()
        #total_samples += len(labels)
    
        if i%10 == 0: print("train_loss = ",b_loss.item())

    #acc = round(sum_correct_pred/total_samples,4)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss

def evaluation(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for images, input_ids, attention_mask in val_data_loader:
            
            #Loading data and labels to device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            #Forward
            dino_features, bert_features = model(images, input_ids, attention_mask)
            #Calculating Loss
            _loss = loss_fn(dino_features, bert_features)
            epoch_loss.append(_loss.item())
            
            # calculate acc per minibatch
            logits,_ = loss_fn.get_logits(dino_features, bert_features)
            labels = loss_fn.get_ground_truth(dino_features.device, dino_features.shape[0])
            sum_correct_pred += (torch.argmax(logits,dim=-1) == labels).sum().item()
            total_samples += len(labels)

    acc = round(sum_correct_pred/total_samples,4)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return 


def train(batch_size, epochs):
    """
    LOAD DATA
    """
    # Define the paths to the dataset and annotations
    i_dir = "'Images/"
   
    # Create the dataset and dataloader
    train_dataset = Meme_DataSet(img_dir = i_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    #eval_dataset = Meme_DataSet(img_dir = i_dir)
    #eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
  
    """
    MODEL. And send it to GPU if we have gpus lol.
    """
    print("\n \t ----------- Model = BertDistilled-DINOv2-CLIP ------------")
    model = clip_for_meme()
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        #model = nn.DataParallel(model,device_ids=[0,1,2,3])
    else:
        device = torch.device("cpu")  
        model = nn.DataParallel(model,device_ids=[0,1,2,3])
        
    #model = nn.DataParallel(model,device_ids=[0,1,2,3]) #unquote when we have multiple gpus my laptop is trash
    model.to(device)
    print("\n \t ----------- Model Loaded------------")
    print("\t *Total Params* = ",sum(p.numel() for p in model.parameters()))
    print("\t *Trainable Params* = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    loss_fn = I_T_ContrastiveLoss(temperature=0.1) # change temperature if needed
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)  #and change learning rate if not working
    print("\n \t ----------- Model Loaded------------")

    for epoch in range(epochs):

        begin = time.time()

        ###Training
        loss = one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        #val_loss, val_acc = val_one_epoch(val_loader, model, loss_fn, device)

        print('\n\t Epoch : ', epoch + 1)
        print("\t Training loss & accuracy: ",round(loss,4))
        print('\t Training time current epoch: ', round((time.time()-begin),2), 'seconds')
        


    torch.save(model.state_dict(),'clip_for_meme.pth')

if __name__=="__main__":
    train(batch_size=16, epochs=3)
