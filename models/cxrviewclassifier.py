#!/usr/bin/env python
# cxrviewclassifier.py
from models.cxrclassifier import *

class CXRViewClassifier(CXRClassifier):
    'A classifier for AP/PA view in chest radiographs'
    def train(self, 
              train_dataset, 
              val_dataset, 
              max_epochs=100, 
              lr=0.01, 
              weight_decay=1e-4,
              batch_size=16,
              early_stopping_rounds=3,
              logpath=None,
              checkpoint_path='checkpoint.pkl',
              verbose=True):
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
            early_stopping_rounds (int): The number of rounds, in which the
                validation loss does not improve from its best value, to wait 
                before stopping training.
            logpath (str): The path at which to write a training log. If None,
                do not write a log.
            checkpoint_path (str): The path at which to save a checkpoint 
                corresponding to the model so far with the best validation loss.
            verbose (bool): If True, print extra information about training.
        Returns:
            model: Trained instance of torch.nn.Module.
        '''
        self.model = torchvision.models.densenet121(pretrained=True)
        self.checkpoint_path = checkpoint_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self._get_optimizer(lr, self.weight_decay)
        # Create torch DataLoaders from the training and validation datasets.
        # Necessary for batching and shuffling data.
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8)
        dataloaders = {
                'train': train_dataloader,
                'val': val_dataloader}

        # Build the model
        self.model = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        # Add a classification head; consists of standard dense layer with
        # sigmoid activation and one output node per pathology in train_dataset
        self.model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 1), 
                torch.nn.Sigmoid())

        # Put model on GPU
        self.model.cuda()

        # Define the optimizer. Use SGD with momentum and weight decay.
        optimizer = self.optimizer
        best_loss = None 
        best_epoch = None 

        # Begin training. Iterate over each epoch to (i) optimize network and
        # (ii) calculate validation loss.
        for i_epoch in range(max_epochs):
            print("-------- Epoch {:03d} --------".format(i_epoch))
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.train(False)

                # Iterate over each batch of data; loss holds a 
                # running sum of the loss from each batch
                loss = 0
                for batch in dataloaders[phase]:
                    inputs, labels, _, appa = batch
                    # batch size may differ from batch_size for the last  
                    # batch in an epoch
                    current_batch_size = inputs.shape[0]

                    # Transfer inputs (images) and labels (arrays of ints) to 
                    # GPU
                    inputs = torch.autograd.Variable(inputs.cuda())
                    labels = torch.autograd.Variable(labels.cuda()).float()
                    appa = torch.autograd.Variable(appa.cuda()).float().view(-1,1)
                    outputs = self.model(inputs)

                    # Calculate the loss
                    optimizer.zero_grad()
                    batch_loss = self.lossfunc(outputs, appa)

                    # If training, update the network's weights
                    if phase == 'train':
                        batch_loss.backward()
                        optimizer.step()

                    # Update the running sum of the loss
                    loss += batch_loss.data.item()*current_batch_size
                dataset_size = len(val_dataset) if phase == 'val' else len(train_dataset)
                loss /= dataset_size
                if phase == 'train':
                    trainloss = loss
                if phase == 'val':
                    valloss = loss

                if phase == 'val':
                    # Check if the validation loss is the best we have seen so
                    # far. If so, record that epoch.
                    if best_loss is None or valloss < best_loss:
                        best_epoch = i_epoch
                        best_loss = valloss
                        self._checkpoint(i_epoch, valloss)
                    # If the validation loss has not improved, decay the 
                    # learning rate
                    else:
                        self.lr /= 10
                        optimizer = self._get_optimizer(
                                 self.lr, 
                                 self.weight_decay)
                        self.optimizer = optimizer

            # Write information on this epoch to a log.
            logstr = "Epoch {:03d}: ".format(i_epoch) +\
                     "training loss {:08.4f},".format(trainloss) +\
                     "validation loss {:08.4f}".format(valloss)
            if not logpath is None:
                with open(logpath, 'a') as logfile:
                    logfile.write(logstr + '\n')
            if verbose:
                print(logstr)

            # If we have gone three epochs without improvement in the validation
            # loss, stop training
            if i_epoch - best_epoch > early_stopping_rounds:
                break
        self.load_checkpoint(self.checkpoint_path)
        return self.model

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
                num_workers=8)
        pred_df = pandas.DataFrame(columns=["path"])
        true_df = pandas.DataFrame(columns=["path"])

        output = numpy.zeros((len(dataset), 1))

        # Iterate over the batches
        for ibatch, batch in enumerate(dataloader):
            inputs, labels, _, appa = batch
            # Move to GPU
            inputs = torch.autograd.Variable(inputs.cuda())
            labels = torch.autograd.Variable(labels.cuda())
            appa = torch.autograd.Variable(appa.cuda())

            true_labels = appa.cpu().data.numpy()
            # Size of current batch. Could be less than batch_size in final 
            # batch
            current_batch_size = true_labels.shape[0]

            # perform prediction
            probs = self.model(inputs).cpu().data.numpy()

            # get predictions and true values for each item in batch
            for isample in range(0, current_batch_size):
                output[batch_size*ibatch + isample] = probs[isample]
        return output
