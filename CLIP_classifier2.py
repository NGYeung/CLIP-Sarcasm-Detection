"""
Created on Thu Aug 7

@author: Yiyang Liu

"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from custom_models.clip_class import clip_for_meme, I_T_ContrastiveLoss, Huber_ContrastiveLoss
from Datasets.CLIP_Datasets import Meme_Classify, Meme_Verify# Custom Dataset
import os, sys
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt



def custom_collate_fn(batch):
    # Find the maximum length in the batch
    #print('image',batch[0]['image'])

    images = [im['image'].squeeze(0) for im in batch]
    inputids = [im['input_ids'].squeeze(0) for im in batch]
    attentionmask = [im['attention'].squeeze(0) for im in batch]
    labels  = torch.tensor([int(im['label']) for im in batch])
 
   
    
    
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
  
    return tensor_images, text_tensor.long(), tensor_attention, labels
    


def one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    classlist = []
    labellist = []

    model.train()
 
    ###Iterating over data loader
    for i, (images, input_ids, attention_mask, labels) in enumerate(train_data_loader):
              
        #Loading data and labels to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

       
        optimizer.zero_grad()
        
        class_ = model(images, input_ids, attention_mask)
   
        class_ = class_.float()
        classlist.append(class_.detach())
        labellist.append(labels.detach().float())
     
        b_loss = loss_fn(class_, labels.float())
       
       

        
        epoch_loss.append(b_loss.item())      
        #Backward
        b_loss.backward() #compute gradients
        
        
        
                    
            # Analyze gradients
        total_norm = 0.0
        max_grad = 0
        if i%10 == 0:
            print(f"gradient stats:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    if grad.abs().mean() > max_grad:
                        max_grad = grad.abs().mean()
                   
                    print(f"{name} - Mean: {grad.abs().mean()}, Max: {grad.max()}, Min:{grad.min()} ,Grad Norm: {param_norm}")
                    
            # Analyze gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad*4)
       
        optimizer.step()
        
        
  

        #labels = torch.arange(logits.shape[0], device=device, dtype=torch.long)
        
        
        sum_correct_pred += (torch.round(class_) == labels).sum().item()
        total_samples += len(labels)

     
        if i%15 == 0: 
            print("train_loss = ",b_loss.item())
      
        
    labellist = torch.cat(labellist)
    classlist = torch.cat(classlist)
    acc = round(sum_correct_pred/total_samples,4)
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    auroc = roc_auc_score( labellist, classlist)
  
    return epoch_loss, auroc, acc


def classification(test_loader, model, loss_fn, device):
    
    
  
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    classlist = []
    labellist = []
    

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for images, input_ids, attention_mask, labels in test_loader:
            prediction = []
            
            
            #Loading data and labels to device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            

            #Forward
            class_ = model(images, input_ids, attention_mask)
            
            classlist.append(class_.detach())
            labellist.append(labels.detach().float())
            #Calculating Loss
            #logits,_ = loss_fn.logits(txt_features, img_features)
            #loss_fn.set_labels(labels)
            class_ = class_.float()
            _loss = loss_fn(class_, labels.float())
            epoch_loss.append(_loss.item())
            #print(torch.argmax(logits,dim=-1))
            sum_correct_pred += (torch.round(class_) == labels).sum().item()
            total_samples += len(labels)
            
 
            
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    labellist = torch.cat(labellist)
    classlist = torch.cat(classlist)
    acc = round(sum_correct_pred/total_samples,4)
    ###Acc and Loss
    auroc = roc_auc_score( labellist, classlist)
    print("AUROC:", auroc)
    return epoch_loss, auroc, acc


def train(batch_size, epochs, ratio = 0.5, load = 0, train_loader = None, loss = 1):
    """
    LOAD DATA ratio =  the ratio of data used for training. loss = 1 =  contractive loss 2 = huber
    """
    # Define the paths to the dataset and annotations
    
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    #i_dir = "Datasets/Images_debug"
    #t_dir = 'Datasets/text_debug.csv'
    i_dir = "Datasets/Images/"
    t_dir = "Datasets/TEXT_sentences.csv"
   
    # Create the dataset and dataloader
    train_dataset = Meme_Classify(img_dir = i_dir, text_file = t_dir)
    subset_idx = np.random.randint(0,len(train_dataset),int(round(len(train_dataset)*ratio)))
    train_size = int(round(len(train_dataset)*ratio)*0.8)
    train_id = subset_idx[0:train_size]
    test_id = subset_idx[train_size:]
    
    
    
    train_subset = Subset(train_dataset, train_id)
    test_subset = Subset(train_dataset, test_id)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    
    #eval_dataset = Meme_DataSet(img_dir = i_dir)
    eval_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
  
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
    
    if loss == 1:
        pass
    loss_fn = nn.BCELoss()
    
   
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)  #and change learning rate if not working
    print("\n \t ----------- Model Loaded------------")
    
    
    if load>0: 
        print('\n \t ------------ loading epoch ', str(load), ' ------------')
        model, optimizer, epoch = load_checkpoint('/scratch/borcea_root/borcea0/yiyangl/model_states/class_checkpoint_'+str(load)+'.pt',model, optimizer)
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
        loss, auroc, acc = one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        #val_loss, val_acc = val_one_epoch(val_loader, model, loss_fn, device)

        print('\n\t Epoch : ', epoch + 1)
        print('\n\t Accuracy : ', acc)
        print('\n\t AUROC : ', auroc)
        print("\t Training loss: ",round(loss,4))
        print('\t Training time current epoch: ', round((time.time()-begin),2), 'seconds')
        
        save_checkpoint(model, optimizer, epoch+1, '/scratch/borcea_root/borcea0/yiyangl/model_states/class_checkpoint_'+str(epoch+1)+'.pt')
        epoch += 1
        
        
    torch.save(model.state_dict(),'clip_for_meme.pth')
    epochloss, auroc, acc = classification(eval_loader, model, loss_fn, device)
    print('prediction loss:', epochloss, ' accuracy: ', acc, 'auroc', auroc)
    
    
    
    
def get_data(train_test_split=0.8):
    '''train_test_split = ratio of data used for training
    '''
    np.random.seed(42)
    torch.manual_seed(42)
    
    #i_dir = "Datasets/Images_debug"
    #t_dir = 'Datasets/text_debug.csv'
    i_dir = "Datasets/Images/"
    t_dir = "Datasets/TEXT_sentences.csv"
   
    # Create the dataset and dataloader
    train_dataset = Meme_Classify(img_dir = i_dir, text_file = t_dir)
    subset_idx = np.random.randint(0,len(train_dataset),int(round(len(train_dataset)*ratio)))
    train_size = int(round(len(train_dataset)*ratio)*train_test_split)
    train_id = subset_idx[0:train_size]
    test_id = subset_idx[train_size:]
    
    
    
    train_subset = Subset(train_dataset, train_id)
    test_subset = Subset(train_dataset, test_id)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=custom_collate_fn)
    
    #eval_dataset = Meme_DataSet(img_dir = i_dir)
    eval_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    
    return train_loader, eval_loader, train_id, test_id
        
        
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






if __name__=="__main__":
    train(batch_size=5, epochs=3)