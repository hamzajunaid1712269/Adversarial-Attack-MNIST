from torchvision import datasets, transforms
from six.moves import urllib
import torch
    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
   ('/'.join([new_mirror, url.split('/')[-1]]), md5)
   for url, md5 in datasets.MNIST.resources
]


def load_dataset(dataset_name):
    classes=10
    channels=1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    dataset = datasets.MNIST(root = './data', train=True, transform = transform, download=True)
    train, val = torch.utils.data.random_split(dataset, [50000, 10000])
    test = datasets.MNIST(root = './data', train=False, transform = transform, download=True)
    return(train,val,test,classes,channels)

