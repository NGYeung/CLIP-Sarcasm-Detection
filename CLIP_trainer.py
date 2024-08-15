"""
Created on Thu Aug 7

@author: Yiyang Liu

"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from custom_models.clip import clip_for_meme, I_T_ContrastiveLoss
from Datasets.CLIP_Datasets import Meme_DataSet # Custom Dataset
import os, sys

import seaborn as sns
import matplotlib.pyplot as plt



def custom_collate_fn(batch):
    # Find the maximum length in the batch
    
    images = [im['image'].squeeze(0) for im in batch]
    inputids = [im['input_ids'].squeeze(0) for im in batch]
    attentionmask = [im['attention'].squeeze(0) for im in batch]
    #labels  = [int(im['label']) for im in batch]]
    
    
    max_len = max(ip.size(0) for ip in inputids)
    feature_size = inputids[0].size(0)
    #feature_size2 = attentionmask[0].size(1)
    # Pad the tensors to have the same size
    padded = []
    pat = []
    for ts in inputids:
        #ts_p = torch.cat([ts, torch.zeros(max_len - ts.size(0), feature_size)], dim=0)
        ts_p = torch.cat([ts, torch.zeros(max_len - ts.size(0))], dim=0)
        padded.append(ts_p)
        
    for att in attentionmask:
        #att_p = torch.cat([att, torch.zeros(max_len - att.size(0), feature_size)], dim=0)
        att_p = torch.cat([att, torch.zeros(max_len - att.size(0))], dim=0)
        pat.append(att_p)
    

    text_tensor = torch.stack(padded, dim=0)
    
    tensor_images = torch.stack(images, dim = 0)
   
    tensor_attention = torch.stack(pat, dim = 0)
    
    #tensor_labels = torch.stack(labels, dim = 0)
     
    #return {'image': tensor_images, 'input_ids': stacked_tensor, 'attention': tensor_attention}
    return tensor_images, text_tensor.long(), tensor_attention
    


def one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    
    model.train()
 
    ###Iterating over data loader
    for i, (images, input_ids, attention_mask) in enumerate(train_data_loader):
              
        #Loading data and labels to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        img_features, txt_features = model(images, input_ids, attention_mask)
        #Calculating Loss
        vis = 0
        if i%10 == 0: 
            vis = 1
        b_loss = loss_fn(img_features, txt_features, visualize = vis)
        epoch_loss.append(b_loss.item())      
        #Backward
        b_loss.backward() #compute gradients
        optimizer.step()

        # calculate acc per minibatch
        text_logits = loss_fn.logits(img_features, txt_features)
        # The following quoted because we don't need labels.
        #labels = loss_fn.get_ground_truth(img_features.device, txt_features.shape[0])
        #sum_correct_pred += (torch.argmax(logits,dim=-1) == labels).sum().item()
        #total_samples += len(labels)
    
        if i%10 == 0: print("train_loss = ",b_loss.item())

    #acc = round(sum_correct_pred/total_samples,4)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, text_logits

def evaluation(model, loss_fn, device, batch_size, ratio = 0.02, test_img_dir="Datasets/Images_test/",test_text_path="Datasets/TEXT_test_sentences.csv"):
    
    
   
    # Create the dataset and dataloader
    test_dataset = Meme_DataSet(img_dir = test_img_dir, text_file = test_text_path)
  
    subset_idx = np.random.randint(0,len(test_dataset),int(round(len(test_dataset)*ratio)))
    test_subset = Subset(test_dataset, subset_idx)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn = custom_collate_fn)
    
    ### Local Parameters
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for images, input_ids, attention_mask in test_loader:
            print('ha! test!')
            
            #Loading data and labels to device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            

            #Forward
            img_features, text_features = model(images, input_ids, attention_mask)
            print(img_features.size(), text_features.size())
            #Calculating Loss
            _loss = loss_fn(img_features, text_features)
            epoch_loss.append(_loss.item())
            
            # calculate acc per minibatch
            logits_t,_ = loss_fn.logits(img_features, text_features)
            
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, logits_t, img_features, text_features


def train(batch_size, epochs, ratio = 0.5, load = 0):
    """
    LOAD DATA ratio =  the ratio of data used for training.
    """
    # Define the paths to the dataset and annotations
    
    
    
    
    #i_dir = "Datasets/Images_debug"
    #t_dir = 'Datasets/text_debug.csv'
    i_dir = "Datasets/Images/"
    t_dir = "Datasets/TEXT_sentences.csv"
   
    # Create the dataset and dataloader
    train_dataset = Meme_DataSet(img_dir = i_dir, text_file = t_dir)
    subset_idx = np.random.randint(0,len(train_dataset),int(round(len(train_dataset)*ratio)))
    train_subset = Subset(train_dataset, subset_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,collate_fn=custom_collate_fn)
    
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
        print('\n \t Running on CPU.')
        device = torch.device("cpu")  
        model = nn.DataParallel(model,device_ids=[0,1,2,3])
        
    #model = nn.DataParallel(model,device_ids=[0,1,2,3]) #unquote when we have multiple gpus my laptop is trash
    

    """
    Train
    """
    
        
    loss_fn = I_T_ContrastiveLoss(temperature=0.1) # change temperature if needed
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)  #and change learning rate if not working
    print("\n \t ----------- Model Loaded------------")
    
    
    if load>0: 
        print('\n \t ------------ loading epoch ', str(load), ' ------------')
        model, optimizer, epoch = load_checkpoint('/scratch/borcea_root/borcea0/yiyangl/model_states/checkpoint_'+str(load)+'.pt',model, optimizer)
        epoch += 1
    else:
        epoch = 0
        
    model.to(device)
    print("\n \t ----------- Model Loaded------------")
    print("\t *Total Params* = ",sum(p.numel() for p in model.parameters()))
    print("\t *Trainable Params* = ",sum(p.numel() for p in model.parameters() if p.requires_grad))
        

    while epoch < epochs:

        begin = time.time()

        ###Training
        loss, logits = one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        #val_loss, val_acc = val_one_epoch(val_loader, model, loss_fn, device)

        print('\n\t Epoch : ', epoch + 1)
        print("\t Training loss: ",round(loss,4))
        print('\t Training time current epoch: ', round((time.time()-begin),2), 'seconds')
        
        save_checkpoint(model, optimizer, epoch+1, '/scratch/borcea_root/borcea0/yiyangl/model_states/checkpoint_'+str(epoch+1)+'.pt')
        epoch += 1
        
        
    torch.save(model.state_dict(),'clip_for_meme.pth')
        
        
def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch



def find_matches(model, image_embeddings, text_embeddings, query, n=9):
    
    
    return




if __name__=="__main__":
    train(batch_size=5, epochs=3)
