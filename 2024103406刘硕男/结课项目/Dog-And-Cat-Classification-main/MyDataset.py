import torchvision
import  torch
import shutil
import PIL.Image as Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.nn.functional import one_hot
from torchvision.io import read_image,ImageReadMode
import os.path
to_tensor = transforms.ToTensor()
class MyDataset(Dataset):

    def __init__(self,root,labels,transform,device,isTrain:bool = True,imgSize:int = -1):
        self.device = device
        self.allImgPaths = []
        self.root = root
        if isTrain:
            self.root = os.path.join(self.root,"train")
        else:
            self.root = os.path.join(self.root,"test")
        self.labels = labels
        for label in self.labels:
            label_path = os.path.join(self.root,label)
            imgNames = os.listdir(label_path)
            index = 0
            for imgName in imgNames:
                if index == imgSize:
                    break
                self.allImgPaths.append((int(label),os.path.join(label_path,imgName)))
                if imgSize!=-1:
                    index = index+1

        self.transform = transform

    def __len__(self):
        return len(self.allImgPaths)
    def __getitem__(self, index):
        label,imgPath = self.allImgPaths[index]

        #img = to_tensor(Image.open(imgPath))
        #print(imgPath)
        img = read_image(imgPath,mode=ImageReadMode.RGB).to(self.device)

        img = self.transform(img)
        #return img,one_hot(torch.tensor(label),num_classes=len(self.labels)).float()
        return img,label



