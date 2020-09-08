#!/usr/bin/env python3
# model.py
import os
import time

import numpy
import pandas
import torch
import torchvision

from datasets import *
import sklearn


from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

from tqdm import *

def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))
        
class AlexNet(Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class CXRClassifier(object):
    'A classifier for various pathologies found in chest radiographs'
    def __init__(self):
        '''
        Create a classifier for chest radiograph pathology.
        '''
        self.lossfunc = torch.nn.BCEWithLogitsLoss()

    def build_model(self, n_labels):
        self.model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        # Add a classification head; consists of standard dense layer with
        # sigmoid activation and one output node per pathology in train_dataset
        self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, n_labels))

        # Put model on GPU
        self.model.cuda()
    
    def build_model_scratch(self, n_labels):
        self.model = AlexNet(n_labels)
        # Put model on GPU
        self.model.cuda()

    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=100, 
              lr=0.01, 
              weight_decay=1e-4,
              batch_size=16,
              logpath=None,
              checkpoint_path='checkpoint.pkl',
              verbose=True,
              scratch_train=False,
              freeze_features=False):
        '''
        Train the classifier to predict the labels in the specified dataset.
        Training will start from the weights in a densenet-121 model pretrained
        on imagenet, as provided by torchvision.
        
        Args:
            train_dataset: An instance of ChestXray14Dataset, MIMICDataset, or 
                CheXpertDataset. Used for training neural network.
            val_dataset: An instance of ChestXray14Dataset, MIMICDataset, or 
                CheXpertDataset. Used for determining when to stop training.
            max_epochs (int): The maximum number of epochs for which to train.
            lr (float): The learning rate to use during training.
            weight_decay (float): The weight decay to use during training.
            batch_size (int): The size of mini-batches to use while training.
            logpath (str): The path at which to write a training log. If None,
                do not write a log.
            checkpoint_path (str): The path at which to save a checkpoint 
                corresponding to the model so far with the best validation loss.
            verbose (bool): If True, print extra information about training.
            scratch_train (bool): If True, train an AlexNet from scratch.
            freeze_features (bool): If True, freeze all layers of the network
                except for the final classifier layers (i.e., use fixed feature
                extractor).
        Returns:
            model: Trained instance of torch.nn.Module.
        '''
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                worker_init_fn=train_dataset.init_worker)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                worker_init_fn=val_dataset.init_worker)

        # Build the model
        if scratch_train:
            self.build_model_scratch(len(train_dataset.labels))
        else:
            self.build_model(len(train_dataset.labels))

        # Freeze weights if desired
        if freeze_features:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.classifier.parameters():
                p.requires_grad = True

        # Define the optimizer. Use SGD with momentum and weight decay.
        self.optimizer = self._get_optimizer(lr, self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                5, # epochs between stepping lr
                gamma=0.1)
        # Begin training. Iterate over each epoch to (i) optimize network and
        # (ii) calculate validation loss.
        best_loss = None 
        best_auroc = None
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            trainloss = self._train_epoch(train_dataloader)
            trainloss /= len(train_dataset)
            valloss, valauroc = self._val_epoch(val_dataloader)
            valloss /= len(val_dataset)
            
            # only save if improvement
            if best_loss is None or valloss < best_loss: 
                best_loss = valloss
                self._checkpoint(i_epoch, valloss, suffix='.best_loss')
            if best_auroc is None or valauroc > best_auroc:
                best_auroc = valauroc
                self._checkpoint(i_epoch, valloss, suffix='.best_auroc')
                
            # If the validation loss has not improved, decay the 
            # learning rate
            scheduler.step()

            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss) + \
                     "validation auroc {:.4f}".format(valauroc)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)
        self.load_checkpoint(self.checkpoint_path+'.best_auroc')
        return self.model

    def _train_epoch(self, train_dataloader):
        self.model.train(True)
        loss = 0
        for batch in tqdm(train_dataloader, leave=False):
            inputs, labels, _, ds = batch
            # batch size may differ from batch_size for the last  
            # batch in an epoch
            current_batch_size = inputs.shape[0]

            # Transfer inputs (images) and labels (arrays of ints) to 
            # GPU
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda()).float()
            outputs = self.model(inputs)

            # Calculate the loss
            self.optimizer.zero_grad()
            batch_loss = self.lossfunc(outputs, labels)

            # update the network's weights
            batch_loss.backward()
            self.optimizer.step()

            # Update the running sum of the loss
            loss += batch_loss.data.item()*current_batch_size
        return loss

    def _val_epoch(self, val_dataloader):
        self.model.train(False)
        loss = 0
        logits_list = []
        true_list = []
        for batch in tqdm(val_dataloader, leave=False):
            inputs, labels, _, ds = batch
            # batch size may differ from batch_size for the last  
            # batch in an epoch
            current_batch_size = inputs.shape[0]

            # Transfer inputs (images) and labels (arrays of ints) to 
            # GPU
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda()).float()
            outputs = self.model(inputs)
            logits_list.append(outputs.cpu().data.numpy())
            true_list.append(labels.cpu().data.numpy())

            # Calculate the loss
            batch_loss = self.lossfunc(outputs, labels)

            # Update the running sum of the loss
            loss += batch_loss.data.item()*current_batch_size
        logits = numpy.concatenate(logits_list, axis=0)
        true = numpy.concatenate(true_list, axis=0)
        odds = numpy.exp(logits)
        probs = odds/(1+odds)
        auroc = sklearn.metrics.roc_auc_score(true[:,-1], probs[:,-1])
        return loss, auroc

    def _checkpoint(self, epoch, valloss, suffix=None):
        '''
        Save a checkpoint to self.checkpoint_path, including the full model, 
        current epoch, learning rate, and random number generator state.
        '''
        state = {'model': self.model,
                 'best_loss': valloss,
                 'epoch': epoch,
                 'rng_state': torch.get_rng_state(),
                 'LR': self.lr ,
                 'optimizer': self.optimizer.state_dict()}
        checkpoint_path = self.checkpoint_path
        if suffix is not None:
            checkpoint_path = checkpoint_path + suffix
        torch.save(state, checkpoint_path)

    def _get_optimizer(self, lr, weight_decay):
        opt = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay)
        return opt

    def load_checkpoint(self, path, load_optimizer=False):
        checkpoint = torch.load(path)
        self.model = checkpoint['model']
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def predict(self, dataset, batch_size=16):
        '''
        Predict the labels of the images in 'dataset'. Outputs indicate the
        probability of a particular label being positive (interpretation 
        depends on the dataset used during training).

        Args:
            dataset: An instance of ChestXray14Dataset, MIMICDataset, or 
                CheXpertDataset.
        Returns:
            predictions (numpy.ndarray): An array of floats, of shape 
                (nsamples, nlabels), where predictions[i,j] indicates the 
                probability of label j of sample i being positive.
        '''
        self.model.train(False)

        # Build a dataloader to batch predictions
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                worker_init_fn=dataset.init_worker)
        pred_df = pandas.DataFrame(columns=["path"])
        true_df = pandas.DataFrame(columns=["path"])

        output = numpy.zeros((len(dataset), len(dataset.labels)))

        # Iterate over the batches
        for ibatch, batch in enumerate(tqdm(dataloader, leave=False)):
            inputs, labels, _, ds = batch
            # Move to GPU
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda())

            true_labels = labels.cpu().data.numpy()
            # Size of current batch. Could be less than batch_size in final 
            # batch
            current_batch_size = true_labels.shape[0]

            # perform prediction
            logits = self.model(inputs)
            logits = logits.cpu().data.numpy()
            odds = numpy.exp(logits)
            probs = odds/(1+odds)

            # get predictions and true values for each item in batch
            for isample in range(0, current_batch_size):
                for ilabel in range(len(dataset.labels)):
                    output[batch_size*ibatch + isample, ilabel] = \
                        probs[isample, ilabel]
        return output

