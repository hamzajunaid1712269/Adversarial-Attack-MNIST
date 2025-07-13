import torch
import numpy as np
import target_models
import os
from torchvision import datasets, transforms
import torch.nn.functional as F
from prepare_dataset import load_dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

def fgsm_attack(input,epsilon,data_grad):
  pert_out = input + epsilon*data_grad.sign()
  pert_out = torch.clamp(pert_out, 0, 1)
  return pert_out

model_name = 'MNISTDEF'


f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join( 'DEFENSE.pth.tar')
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()


def test1(model,device,test_loader,epsilon,attack):
  print("work")
  correct = 0
  adv_examples = []
  y_pred = []
  y_true = []

  for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      
      data.requires_grad = True
      output = model(data)
      init_pred = output.max(1, keepdim=True)[1] 
      if init_pred.item() != target.item():
          continue
      loss = F.nll_loss(output, target)
      model.zero_grad()
      loss.backward()
      data_grad = data.grad.data
     
      
      perturbed_data = fgsm_attack(data,epsilon,data_grad)
      
      
      
      output = model(perturbed_data)
      
      
      target = target.data.cpu().numpy()
      y_true.extend(target)
      


      final_pred = output.max(1, keepdim=True)[1]
      outputD = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
      y_pred.extend(outputD)
      print("predict",y_pred)
      print("target",target)
      
      if final_pred.item() == target.item():
          correct += 1
  classes = ('0','1', '2', '3', '4', '5',
        '6', '7', '8', '9')
  cf_matrix = confusion_matrix(y_true, y_pred)
  df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
  plt.figure(figsize = (12,7))
  sns.heatmap(df_cm, annot=True)
  plt.savefig('output.png')


  final_acc = correct/float(len(test_loader))
  print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

  return final_acc, adv_examples
dataset_name = "mnist"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data,val_data,test_data, in_channels, num_classes = load_dataset(dataset_name)
   
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
test1(f,device,test_loader,0.1,'fgsm')


