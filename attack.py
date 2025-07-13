from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import target_models
import torch.nn as nn
from torch.autograd import Variable

import cv2
import numpy as np
import os

import cv2
win = Tk()
global x





model_name = 'MNISTDEF'
img_path   = 'images/7.jpg'
   
device='cpu'
f = getattr(target_models, model_name)(1, 10)
checkpoint_path_f = os.path.join( 'DEFENSE.pth.tar')
checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
f.load_state_dict(checkpoint_f["state_dict"])
f.eval()
    

orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = orig.copy().astype(np.float32)
img = img[None, None, :, :]/255.0
criterion = nn.NLLLoss()

x = Variable(torch.from_numpy(img), requires_grad=True)


out = f(x)
pred = np.argmax(out.data.cpu().numpy())






alpha = 100
eps=100
num_iter=100
    

   
 
break_loop = False
for i in range(num_iter):

        if break_loop == False:

            inp = Variable(torch.from_numpy(img), requires_grad=True)
            out = f(inp)
           
            loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

            loss.backward()

            # this is the method
            perturbation = (alpha/255.0) * torch.sign(inp.grad.data)
            perturbation = torch.clamp((inp.data + perturbation) - orig, min=-eps/255.0, max=eps/255.0)
            inp.data =  x + perturbation

            inp.grad.data.zero_()
            ################################################################

pred_adv,y = torch.max(F.softmax(f(inp), 1), 1)
print(y)
print(pred_adv)
           
