import argparse
import json
import numpy as np
import os
import torch
from  tensorboardX import SummaryWriter
 
from datetime import datetime
from pathlib import Path
from sklearn import metrics

from loader import load_data
from model import Net
from model import NetFactory
from model import count_parameters
from model import count_trainable_parameters
#from model_alexnet import Net
from run_model import run_model
import warnings
warnings.filterwarnings('ignore')

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(rundir, model_name, epochs, learning_rate, augment,weight_decay, patience, factor, use_gpu):
    train_loader, valid_loader, test_loader = load_data(augment, use_gpu)
    
    writer = SummaryWriter(rundir)
    model = NetFactory.createNet(model_name)
    print("total parameters: " , count_parameters(model))
    print("trainable parameters: " , count_trainable_parameters(model))
    
    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, threshold=1e-4)

    best_val_loss = float('inf')

    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(writer, epoch, model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(writer, epoch, model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'{model_name}_val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) /'models'/file_name
            torch.save(model.state_dict(), save_path)
            save_path2 = Path(rundir) /'models'/'bestmodel'
            torch.save(model.state_dict(), save_path2)

        lr = get_learning_rate(optimizer)
        if len(lr) > 0 : writer.add_scalar('data/learing_rate', lr[0], epoch)

        writer.add_scalar('data/train_loss', train_loss, epoch) 
        writer.add_scalar('data/train_auc', train_auc, epoch) 
        writer.add_scalar('data/val_loss', val_loss, epoch) 
        writer.add_scalar('data/val_auc', val_auc, epoch) 

    writer.export_scalars_to_json(rundir+ "/loss_auc.json") 
    writer.close()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available() 
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)
    os.makedirs(Path(args.rundir)/'models', exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(args.rundir, args.model, args.epochs, args.learning_rate, args.augment, args.weight_decay, args.max_patience, args.factor, use_gpu)
