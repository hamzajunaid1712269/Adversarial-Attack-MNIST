import torch.nn as nn
import torch
import torch.nn.functional as F




class MNISTConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MNISTConvNet, self).__init__()

       
        input_channels = 1

        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(4 * 4 * 64, 200)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        

    def forward(self, x):
        out = self.conv32(x)
        out = self.conv64(out)
        out = out.view(-1, 4 * 4 * 64)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out
    

class MNISTDEF(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MNISTDEF, self).__init__()
   
        input_channels = 1

        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv64 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(4 * 4 * 32, 100)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
   
        out = self.conv32(x)
        out = self.conv64(out)
        out = out.view(-1, 4 * 4 * 32)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out
