import numpy as np
import os
import pickle
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy.ndimage.interpolation import zoom

INPUT_DIM = 224
INPUT_DIM_Z = 150
#INPUT_DIM = 86
#INPUT_DIM_Z = 66
MAX_PIXEL_VAL = 255
MEAN = 59
STDDEV = 48

class Dataset(data.Dataset):
    def __init__(self, datadirs, augment=False, use_gpu=False):
        super().__init__()
        self.augment = augment
        self.use_gpu = use_gpu 

        label_dict = {}
        self.paths = []

        for i, line in enumerate(open('metadata.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            subject = line[0]
            label = line[1]
            label_dict[subject] = int(int(label) > 0)

        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir+'/'+file)

        self.labels = [label_dict[path[6:19]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]

        img = nib.load(path)
        vol = img.get_fdata()

        # crop middle
        pad = int((vol.shape[1] - INPUT_DIM)/2)
        pad_z = int((vol.shape[2] - INPUT_DIM_Z)/2)
        vol = vol[pad:-pad,pad:-pad,pad_z:-pad_z]
        
        # swap axis
        vol = np.swapaxes(vol,0,2) 

        # transform 
        if self.augment is True:
            if np.random.choice([True, False]):
               scale = np.random.uniform(0.9, 1.1)
               vol = zoom(vol, scale)
            if np.random.choice([True, False]):
               angle = np.random.randint(-10, 10)
               vol = rotate(vol, angle, axes=(1, 2), reshape=True, output=None, order=3, mode='constant', cval=0.0, prefilter=True)
            if np.random.choice([True, False]):
               shiftx = np.random.randint(-10, 10)
               shifty = np.random.randint(-10, 10)
               vol = shift(vol, (0, shiftx, shifty), cval=0) 
            if np.random.choice([False, False]):  #flip doesn't help, let's turn off it
               vol = np.flip(vol,0)
            if np.random.choice([False, False]): #add noise doesn't help, let's turn off it
               mu, sigma = 0, 0.10
               noise = np.random.normal(mu, sigma, vol.shape)
               vol = vol + noise
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV

        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)


def load_data(augment=False, use_gpu=False):
    #train_dirs = ['vol08','vol04','vol03','vol09','vol06','vol07', 'vol10', 'vol11', 'vol12', 'vol13', 'vol01',
    #              'vo208','vo204','vo203','vo209','vo206','vo207', 'vo210', 'vo211', 'vo212', 'vo213', 'vo201']
    train_dirs = ['vol08','vol04','vol03','vol09','vol06','vol07', 'vol10', 'vol11', 'vol12', 'vol13', 'vol01']
    valid_dirs = ['vol05', 'vol14', 'vol16', 'vol02','vol15']
    test_dirs = ['vol05', 'vol14', 'vol16', 'vol02','vol15']
    #valid_dirs = ['vol05', 'vol14', 'vol16', 'vol02','vol15']
    #test_dirs = ['vol05', 'vol14', 'vol16', 'vol02','vol15']
    #train_dirs = [ 'vol12', 'vol13']
    #valid_dirs = ['vol05', 'vol14']
    #test_dirs = ['vol01','vol02','vol15']

    #train_dataset = Dataset(train_dirs, train_transform, use_gpu)
    train_dataset = Dataset(train_dirs, augment, use_gpu)
    valid_dataset = Dataset(valid_dirs, False, use_gpu)
    test_dataset = Dataset(test_dirs, False, use_gpu)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=2, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=2, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    train_loader, valid_loader, test_loader = load_data(True, True)


