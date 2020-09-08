#!/usr/bin/env python
import io

import h5py
import numpy
from PIL import Image

from datasets.padchestdataset import PadChestDataset
from datasets.cxrdataset import H5Dataset

class PadChestH5Dataset(PadChestDataset, H5Dataset):
    '''
    HDF5 dataset for PadChest images.
    '''
    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493,
            labels='CheXpert',
            initialize_h5=False,
            pneumo=None):
        '''
        initialize_h5: (bool) If true, open a handle for the HDF5 file when the
            class is instantiated. Use `False` when the dataset will be wrapped
            by a PyTorch dataloader with num_workers > 0, and True otherwise.
        '''
        self.h5path = "data/PadChest/padchest.h5"
        if initialize_h5:
            self.h5 = h5py.File(self.h5path, 'r', swmr=True)
        super().__init__(fold, 
                         include_lateral=include_lateral, 
                         random_state=random_state, 
                         labels=labels,
                         pneumo=pneumo)  

    def init_worker(self, worker_id):
        self.h5 = h5py.File(self.h5path, 'r', swmr=True)

    def _raw_image_from_disk(self, idx):
        '''
        Retrieve the raw PIL image from storage.
        '''
        imagename = self.df.ImageID.iloc[idx]
        data = self.h5['images'].get(imagename)
        image = Image.open(io.BytesIO(numpy.array(data)))
        return image
