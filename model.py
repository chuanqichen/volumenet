from __future__ import generators
import torch
import torch.nn as nn
from torchvision import models
import warnings
warnings.filterwarnings('ignore')

class NetFactory:
    factories = {}
    def addFactory(name, netFactory):
        NetFactory.factories.put[name] = netFactory
    addFactory = staticmethod(addFactory)
    # A Template Method:
    def createNet(name):
        if name not in NetFactory.factories:
            NetFactory.factories[name] = \
              eval(name + '.Factory()')
        return NetFactory.factories[name].create()
    createNet = staticmethod(createNet)

class Net(nn.Module): pass

def netNameGen(n):
    types = Net.__subclasses__()
    for i in range(n):
        yield random.choice(types).__name__

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() )

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class SqueezeNet(Net):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.model = models.squeezenet1_1(pretrained=True)
        for param in self.model.parameters():
           param.requires_grad = False
        ct = 0; 
        for child in self.model.children():
           ct += 1
           if ct > 6:
             for param in child.parameters():
                 param.requires_grad = True

        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

    class Factory:
        def create(self): return SqueezeNet()

class VGG19(Net):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.model =  models.vgg19(pretrained=True)
        for param in self.model.parameters():
           param.requires_grad = False
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 1)

    def forward(self,x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.dropout(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

    class Factory:
        def create(self): return VGG19()


class AlexNet(Net):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.model = models.alexnet(pretrained=True)
        for param in self.model.parameters():
           param.requires_grad = False
        #self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)
        """
        ct = 0
        for child in self.model.children():
           ct += 1
           if ct == 11:
             for param in child.parameters():
                 param.requires_grad = True
        """

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        #x = self.dropout(x)
        #x = self.conv1(x)
        x = self.dropout(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

    class Factory:
        def create(self): return AlexNet()

class Resnet(Net):
    def __init__(self):
        super().__init__()
    
        model = models.resnet18(pretrained = True)
        num_final_in = model.fc.in_features
        for param in model.parameters():
           param.requires_grad = False

        model.fc = nn.Linear(num_final_in, 300)
        
        self.vismodel = nn.Sequential(*list(model.children()))
        self.projective = nn.Linear(512,400)
        self.nonlinearity = nn.ReLU(inplace=True)
        self.projective2 = nn.Linear(400,1)
    
    def forward(self,x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.vismodel(x)
        x = torch.squeeze(x)
        x = self.projective(x)
        x = self.nonlinearity(x)
        x = self.projective2(x)
        return x

    class Factory:
        def create(self): return Resnet()
