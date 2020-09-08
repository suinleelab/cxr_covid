#!/usr/bin/env python3
import os 

import numpy
import torch
from torchvision import transforms
from PIL import Image

# use imagenet mean,std for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class CXRDataset(torch.utils.data.Dataset):
    '''
    Base class for chest radiograph datasets.
    '''
    # define torchvision transforms as class attribute
    _transforms = {
        'train': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    _transforms['test'] = _transforms['val']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self._raw_image_from_disk(idx)
        if self.transform:
            image = self.transform(image)
        label = self._get_label(idx)

        return (image, label, self.df.index[idx], ['None'])

    def init_worker(self, worker_id):
        pass

    def _raw_image_from_disk(self, idx):
        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')
        return image

    def _get_label(self, idx):
        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A":
                if(self.df[self.labels[i].strip()].iloc[idx].astype('int') > 0):
                    label[i] = self.df[self.labels[i].strip()
                                       ].iloc[idx].astype('int')
        return label

    def get_all_labels(self):
        '''
        Return a numpy array of shape (n_samples, n_dimensions) that includes 
        the ground-truth labels for all samples.
        '''
        ndim = len(self.labels)
        nsamples = len(self)
        output = numpy.zeros((nsamples, ndim))
        for isample in range(len(self)):
            output[isample] = self._get_label(isample)
        return output

class H5Dataset(CXRDataset):
    _transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    _transforms['test'] = _transforms['val']
