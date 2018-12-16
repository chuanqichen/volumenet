from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from  tensorboardX import SummaryWriter
from torch.autograd import Variable
from loader import load_data

def precision_recall(writer, step, labels, preds):

    average_precision = average_precision_score(labels, preds)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    precision, recall, _ = precision_recall_curve(labels, preds)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    fig = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
    plt.savefig('Precision-Recallcurve:AP={0:0.2f}.png'.format(
          average_precision))

    writer.add_figure("figure", fig, step, close=True, walltime=None)

def run_model(writer, step, model, loader, train=False, optimizer=None):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        vol, label = batch
        if loader.dataset.use_gpu:
            vol = vol.cuda()
            label = label.cuda()
        vol = Variable(vol)
        label = Variable(label)

        logit = model.forward(vol)

        loss = loader.dataset.weighted_loss(logit, label)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()

        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    if not train :
       precision_recall(writer, step, labels, preds)

    #cm_lr = metrics.confusion_matrix(fpr, tpr)
    #report_lr = metrics.precision_recall_fscore_support(fpr, tpr, average='binary')
    #precision = report_lr[0]
    #recall = report_lr[1]
    #f1 = report_lr[2]
    #writer.add_pr_curve("", fpr, tpr, step, threshold)
    #writer.add_pr_curve("", labels, preds, step, threshold)

    return avg_loss, auc, preds, labels
