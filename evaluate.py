from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from  tensorboardX import SummaryWriter
import argparse

from torch.autograd import Variable
from model import Net
from model import NetFactory
from loader import load_data
from run_model import run_model
import warnings
warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--model', type=str, required=True)
    return parser

def evaluate(split, model_name, model_path, augment, use_gpu):
    train_loader, valid_loader, test_loader = load_data(augment, use_gpu)

    writer = SummaryWriter()
    model = NetFactory.createNet(model_name)
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(writer, 1, model, loader)

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')

    return preds, labels

if __name__ == '__main__':
    args = get_parser().parse_args()
    use_gpu = torch.cuda.is_available()
    evaluate(args.split, args.model, args.model_path, args.augment, use_gpu)
