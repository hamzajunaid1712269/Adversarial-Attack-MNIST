from prepare_dataset import load_dataset
import target_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.autograd import Variable

import matplotlib.pyplot as plt

import numpy as np



import time
import shutil
import cv2


np.random.seed(42)
torch.manual_seed(42)



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train(model,device,optimizer,scheduler,criterion,train_loader,val_loader,Temp,epochs):
  data_loader = {'train':train_loader,'val':val_loader}
  print("Fitting the model...")
  train_loss,val_loss=[],[]
  for epoch in range(epochs):
    loss_per_epoch,val_loss_per_epoch=0,0
    for phase in ('train','val'):
      for i,data in enumerate(data_loader[phase]):
        input,label  = data[0].to(device),data[1].to(device)
        
        output = model(input)
        output = F.log_softmax(output/Temp,dim=1)
        #calculating loss on the output
        loss = criterion(output,label)
        if phase == 'train':
          optimizer.zero_grad()
          #grad calc w.r.t Loss func
          loss.backward()
          #update weights
          optimizer.step()
          loss_per_epoch+=loss.item()
        else:
          val_loss_per_epoch+=loss.item()
    scheduler.step(val_loss_per_epoch/len(val_loader))
    print("Epoch: {} Loss: {} Val_Loss: {}".format(epoch+1,loss_per_epoch/len(train_loader),val_loss_per_epoch/len(val_loader)))
    train_loss.append(loss_per_epoch/len(train_loader))
    val_loss.append(val_loss_per_epoch/len(val_loader))
  return train_loss,val_loss



def defense(model_name,defense_name,device,train_loader,val_loader,epochs,Temp):
  modelN = getattr(target_models, model_name)(in_channels, num_classes)
  modelN.to(device);
 
  optimizerN = optim.Adam(modelN.parameters(),lr=0.0001, betas=(0.9, 0.999))
  schedulerN = optim.lr_scheduler.ReduceLROnPlateau(optimizerN, mode='min', factor=0.1, patience=3)
  modelD = getattr(target_models, defense_name)(in_channels, num_classes)
  modelD.to(device);
   
  optimizerD= optim.Adam(modelD.parameters(),lr=0.0001, betas=(0.9, 0.999))
  schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.1, patience=3)

  criterion = nn.NLLLoss()
  
  lossF,val_lossF=train(modelN,device,optimizerN,schedulerN,criterion,train_loader,val_loader,Temp,epochs)
  fig = plt.figure(figsize=(5,5))
  plt.plot(np.arange(1,epochs+1), lossF, "*-",label="Loss")
  plt.plot(np.arange(1,epochs+1), val_lossF,"o-",label="Val Loss")
  plt.title("Network ")
  plt.xlabel("Num of epochs")
  plt.legend()
  plt.show()

  
  for data in train_loader:
    input, label  = data[0].to(device),data[1].to(device)
    softlabel  = F.log_softmax(modelN(input),dim=1)
    data[1] = softlabel
    
  lossF1,val_lossF1=train(modelD,device,optimizerD,schedulerD,criterion,train_loader,val_loader,Temp,epochs)
  torch.save({
    'state_dict': modelD.state_dict(),
    'optimizer' : optimizerD.state_dict(),
}, 'DEFENSE.pth.tar')
  fig = plt.figure(figsize=(5,5))
  plt.plot(np.arange(1,epochs+1), lossF1, "*-",label="Loss")
  plt.plot(np.arange(1,epochs+1), val_lossF1,"o-",label="Val Loss")
  plt.title("Network Defense'")
  plt.xlabel("Num of epochs")
  plt.legend()
  plt.show() 
  
  







if __name__ == '__main__':

    
    dataset_name = "mnist"
    model_name = "MNISTConvNet"
    defense_name="MNISTDEF"
  
    epoch = 10
    batch_size = 1
    
    seed = 0
    



    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data,val_data,test_data, in_channels, num_classes = load_dataset(dataset_name)
   
    
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=1,shuffle=True)
    
    
    
  
    
    
    defense(model_name,defense_name,device,train_loader,val_loader,epoch,100)
