import numpy as np 
import pandas as pd 
import os
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.datasets as td
from torch.autograd import Variable
from PIL import Image
import pickle
import torch.optim as optim

hparams={'start_epoch':8,
        'lr':1e-4,
        'lr_decay':0.9,
        'weight_decay':1e-4,
        'optim':'adam',
        'train_batch_size':200,
        'test_batch_size':100,
        'classes':10,
        'best_acc':0}


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()

        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.shortcut=nn.Sequential()
        if stride!=1 and in_channels!=out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(out_channels))

    def forward(self,x):
        o=F.relu(self.bn1(self.conv1(x)))
        o=self.bn2(self.conv2(o))
        o+=self.shortcut(x)
        o=F.relu(o)
        return o

class Network(nn.Module):
    def __init__(self,num_classes=hparams['classes']):
        super(Network,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)

        self.block1=self.block_creation(64,64,1)
        self.block2=self.block_creation(64,128,2)
        self.block3=self.block_creation(128,256,2)
        self.block4=self.block_creation(256,512,2)

        self.fc=nn.Linear(512,num_classes)

    def block_creation(self,in_channels,out_channels,stride):
        return nn.Sequential(ResidualBlock(in_channels,out_channels,stride),ResidualBlock(out_channels,out_channels,1))    
    
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.block1(out)
        out=self.block2(out)
        out=self.block3(out)
        out=self.block4(out)
        out=nn.AvgPool2d(4)(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out


def train(dataloader,model,criterion,optimiser):
    model.train()
    print("Model Training.....")
    for epoch in range(hparams['start_epoch'],hparams['start_epoch']+200):
        train_loss=0 ; correct=0 ; print("Epoch:{}".format(epoch+1))
        for idx,data in enumerate(dataloader):
            image,label=data
            image=Variable(image).cuda()
            label=Variable(label).cuda()
            optimiser.zero_grad()
            preds=model(image)
            loss=criterion(preds,label)
            loss.backward()
            optimiser.step()
            pred_labels=preds.max(1)[1]
            train_loss+=loss.item()
            correct+=pred_labels.eq(label).sum().item()
            if ((idx+1)%hparams['train_batch_size'])==0:
                print("Counter_id:{},train_loss:{}".format(idx+1,loss.item()))
        torch.save(model.state_dict(),os.path.join('./models',"Epoch_"+str(epoch+1)+'.pth.tar'))

def test(dataloader):
    for epoch in range(hparams['start_epoch'],hparams['start_epoch']+200):
        print(epoch+1)
        models_path=os.path.join('./models',"Epoch_"+str(epoch+1)+'.pth.tar')
        model.load_state_dict(torch.load(models_path))
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                image, label= data
                image=Variable(image).cuda()
                label=Variable(label).cuda()
                preds=model(image)
                outputs_pred=preds.data.max(1)[1]
                correct+=outputs_pred.eq(label).sum().item()
                total+=label.shape[0]

        accuracy=100*(correct)/total
        if accuracy>hparams['best_acc']:
            hparams['best_acc']=accuracy
            #print("Saving...")
            #state={'Model':model.state_dict(),'Accuracy':accuracy,'Epoch':epoch}
            #if not os.path.exists('/checkpoints'):
                #os.mkdir('/checkpoints')
            #torch.save(state,os.path.join('./checkpoints/ckpt.pth'))

    print(accuracy,total)



if __name__=='__main__':
    cifar_mean=(0.4914, 0.4822, 0.4465); cifar_std=(0.2023, 0.1994, 0.2010)
    train_transform=transforms.Compose([transforms.RandomCrop(32,padding=4),
                                        transforms.RandomHorizontalFlip(p=0.6),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=cifar_mean,std=cifar_std)])

    test_transform=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=cifar_mean,std=cifar_std)])

    train_data=td.CIFAR10(root='./data_cifar/',download=True,train=True, transform=train_transform)
    train_dataloader=torch.utils.data.DataLoader(dataset=train_data,batch_size=hparams['train_batch_size'],shuffle=True,num_workers=2)

    test_data=td.CIFAR10(root='./data_cifar/',download=True,train=False, transform=test_transform)
    test_dataloader=torch.utils.data.DataLoader(dataset=test_data,batch_size=hparams['test_batch_size'],shuffle=False,num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model=(Network(num_classes=hparams['classes']))
    model=torch.nn.parallel.DataParallel(model).cuda()

    if hparams['optim']=='adam': optimiser=optim.Adam(model.parameters(),lr=hparams['lr'],weight_decay=hparams['weight_decay'])
    else: optimiser=optim.SGD(model.parameters(),lr=hparams['lr'],weight_decay=hparams['weight_decay'],momentum=0.9)

    criterion = nn.CrossEntropyLoss().cuda()

    if not os.path.exists('./models'):
        os.mkdir('./models')

    train(train_dataloader,model,criterion,optimiser)
    test(test_dataloader)
    



