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



def open_img():
    open_img.x = openfilename()
    open_img.im = Image.open(open_img.x)
    open_img.img = open_img.im.resize((250, 250), Image.ANTIALIAS)
    open_img.img = ImageTk.PhotoImage(open_img.img)
    panel = Label(win, image = open_img.img) 
    panel.image = open_img.img
    #panel.grid(row = 2)
    panel.place(x=170, y=0)
    

def openfilename():
    filename = filedialog.askopenfilename(title ='"pen')
    return filename  


def evaluate():
    model_name = 'MNISTConvNet'
    img_path   = openfilename()
    open_img.x = openfilename()
    open_img.im = Image.open(open_img.x)
    open_img.img = open_img.im.resize((100, 100), Image.ANTIALIAS)
    open_img.img = ImageTk.PhotoImage(open_img.img)
    panel = Label(win, image = open_img.img) 
    panel.image = open_img.img
    #panel.grid(row = 2)
    panel.place(x=170, y=0)

    f = getattr(target_models, model_name)(1, 10)
    checkpoint_path_f = os.path.join( 'best_MNISTConvNet_mnist.pth.tar')
    checkpoint_f = torch.load(checkpoint_path_f, map_location='cpu')
    f.load_state_dict(checkpoint_f["state_dict"])
    f.eval()
    def fgsm_attack(input,epsilon,data_grad):
        pert_out = input + epsilon*data_grad.sign()
        pert_out = torch.clamp(pert_out, 0, 1)
        return pert_out

    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = orig.copy().astype(np.float32)

    img = img[None, None, :, :]/255.0
    criterion = nn.NLLLoss()


    device =  'cpu'

    x = Variable(torch.from_numpy(img), requires_grad=True)
    out = f(x)
    pred = np.argmax(out.data.cpu().numpy())
    loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))

    loss.backward()

    adv_img=fgsm_attack(x.data,0.3,x.grad.data)


    prob, y= torch.max(F.softmax(f(x), 1), 1)
    adv ,y_adv=torch.max(F.softmax(f(adv_img), 1), 1)
    
    top = Toplevel()
    top.title("About this application...")

    msg = Message(top, text='Prediction : %d [Prob: %0.4f] %d [Prob: %0.4f] '%(y.item(), prob.item(),y_adv.item(),adv.item())
    )
    msg.pack()
    
    
    print('Prediction : %d [Prob: %0.4f]'%(y.item(), prob.item()))
    print('Prediction : %d [Prob: %0.4f]'%(y_adv.item(), adv.item()))
 
win.title("Colour detection in image")
win.geometry("550x400")
win.configure(background="light blue")
win.resizable(width = True, height = True)

btn1 = Button(win, text ='RUN', command = evaluate,  font=("Calibri 12")).place(x=230, y=320)


win.mainloop()
