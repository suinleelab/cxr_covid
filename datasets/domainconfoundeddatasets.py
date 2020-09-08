#!/usr/bin/env python
import numpy
import torch

class DomainConfoundedDataset(torch.utils.data.Dataset):
    def __init__(self, datasetclass1, datasetclass2):
        self.ds1 = datasetclass1
        self.ds2 = datasetclass2
        self.len1 = len(self.ds1)
        self.len2 = len(self.ds2)
        self.labels = self.ds1.labels
    def init_worker(self, worker_id):
        self.ds1.init_worker(worker_id)
        self.ds2.init_worker(worker_id)
    def __getitem__(self, idx):
        if idx < self.len1:
            item = self.ds1[idx]
        else: # if idx >= self.len1
            item = self.ds2[idx-self.len1]
        return item[0], item[1], item[2], item[3]
    def __len__(self):
        return self.len1+self.len2
    def get_all_labels(self):
        labels1 = self.ds1.get_all_labels()
        labels2 = self.ds2.get_all_labels()
        if labels1.shape[0] == 1:
            return numpy.concatenate((self.ds1.get_all_labels(),
                                      self.ds2.get_all_labels()), axis=1)
        else:
            return numpy.concatenate((self.ds1.get_all_labels(),
                                      self.ds2.get_all_labels()), axis=0)
