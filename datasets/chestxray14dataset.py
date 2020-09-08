#!/usr/bin/env python
import os

import numpy
import pandas
import sklearn.model_selection
from PIL import Image
import random

from datasets.cxrdataset import CXRDataset

def grouped_split(dataframe, metadataframe, random_state=None, test_size=0.05):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by _get_patient_id to return the unique patient identifiers.
    '''
    groups = _get_unique_patient_ids(dataframe)
    traingroups, testgroups = sklearn.model_selection.train_test_split(
            groups,
            random_state=random_state,
            test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    traindict = {}
    testdict = {}
    for idx, row in dataframe.iterrows():
        patient_id = _get_patient_id(idx)
        if patient_id in traingroups:
            traindict[idx] = row
        elif patient_id in testgroups:
            testdict[idx] = row
    traindf = pandas.DataFrame.from_dict(
        traindict, 
        orient='index',
        columns=dataframe.columns)
    testdf = pandas.DataFrame.from_dict(
        testdict, 
        orient='index',
        columns=dataframe.columns)
    trainmetadf = metadataframe.loc[traindf.index,:]
    testmetadf = metadataframe.loc[testdf.index,:]
    
    return traindf, testdf, trainmetadf, testmetadf


def _get_patient_id(path):
    return path.split('_')[0]

def _get_unique_patient_ids(dataframe):
    ids = list(dataframe.index)
    ids = [_get_patient_id(i) for i in ids]
    ids = list(set(ids))
    ids.sort()
    return ids

def _convert_dataframe(df):
    '''
    Convert the labels in 'Data_Entry_2017.csv' to one-hot encoded labels and
    return a new dataframe.
    '''
    columns = ['Image Index',
               'Atelectasis',
               'Cardiomegaly',
               'Consolidation',
               'Edema',
               'Effusion',
               'Emphysema',
               'Fibrosis',
               'Hernia',
               'Infiltration',
               'Mass',
               'Nodule',
               'Pleural_Thickening',
               'Pneumonia',
               'Pneumothorax']
    df.set_index('Image Index', inplace=True)

    new_df_dict = {}
    for irow, row in df.iterrows():
        findings_one_hot = []
        row_findings = row['Finding Labels'].split('|')
        for finding in columns[1:]:
            if finding in row_findings:
                findings_one_hot.append(1)
            else:
                findings_one_hot.append(0)
        new_df_dict[irow] = findings_one_hot

    new_df = pandas.DataFrame.from_dict(
            new_df_dict,
            orient='index',
            columns=columns[1:])
    new_df['COVID'] = numpy.zeros(new_df.shape[0])

    return new_df

class ChestXray14Dataset(CXRDataset):
    def __init__(
            self,
            fold,
            random_state=30493,
            labels='ChestX-ray14',
            pneumo=None,
            subsample_test=False):
        '''
        Create a dataset of the CheXPert images for use in a PyTorch model.

        Args:
            fold (str): The shard of the CheXPert data that the dataset should
                contain. One of either 'train', 'val', or 'test'. The 'test'
                fold corresponds to the images specified in 'valid.csv' in the 
                CheXPert data, while the the 'train' and 'val' folds
                correspond to disjoint subsets of the patients in the 
                'train.csv' provided with the CheXpert data.
            random_state (int): An integer used to see generation of the 
                train/val split from the patients specified in the 'train.csv'
                file provided with the CheXpert dataset. Used to ensure 
                reproducability across runs.
            labels (str): One of either 'CheXpert' or 'ChestX-ray14'. In either
                case, each label will be a boolean array where each element of 
                the array corresponds to a pathology, 1 indicates a 'positive 
                mention' of the pathology, and 0 indicates any of 'at least one 
                uncertain mention with no positive mentions', 'a negative 
                mention', or 'no mention'. If 'CheXpert', the labels will 
                include only pathologies specified by both the CheXpert labeler,
                and in the Chest-Xray14 dataset, i.e.,

                    0:  N/A 
                    1:  Cardiomegaly
                    2:  N/A 
                    3:  N/A 
                    4:  Edema
                    5:  Consolidation
                    6:  Pneumonia
                    7:  Atelectasis
                    8:  Pneumothorax
                    9:  Effusion ("Pleural effusion in CheXpert)
                    10: N/A 
                    11: N/A
                    12: N/A               

                If 'ChestX-ray14', the labels will include all pathologies 
                specified in the ChestX-ray14 dataset, ie.,

                    0:  Atelectasis
                    1:  Cardiomegaly
                    2:  Consolidation
                    3:  Edema
                    4:  Effusion
                    5:  Emphysema
                    6:  Fibrosis
                    7:  Hernia
                    8:  Infiltration
                    9:  Mass
                    10: Nodule
                    11: Pleural thickening
                    12: Pneumonia
                    13: Pneumothorax

                where presence of 'N/A' labels and the order of the pathologies
                is chosen for compatibility with classifiers trained on the 
                ChestX-ray14 data.
        '''

        self.fold = fold

        self.transform = self._transforms[fold]
        self.path_to_images = "data/ChestX-ray14/images_224x224/"
        self.has_appa = False
        self.pneumo = pneumo

        # Load files containing labels, and perform train/valid split if necessary
        labelpath = os.path.join(
             self.path_to_images,
             '../labels/Data_Entry_2017.csv')

        # read in the csv file with labels for all the images
        self.df = pandas.read_csv(labelpath)
        self.meta_df = self.df
        self.df = _convert_dataframe(self.df)
        if self.fold == 'train' or self.fold == 'val':
            # Path to the file containing a list of all images in the train and
            # test folds
            foldpath = os.path.join(
                    self.path_to_images,
                    '../labels/train_val_list.txt')
            # Read the file and convert to a list of strings, e.g., 
            # ['00000001_000.png', '00000001_001.png', ...]
            folddf = pandas.read_csv(foldpath, names=["Image Index"])
            foldlist = folddf["Image Index"].values.tolist()
            # Now select only the portion of self.df (which contains class labels)
            # that is found in 'foldlist'
            self.df = self.df.loc[foldlist, :]
            self.meta_df = self.meta_df.loc[foldlist, :]
            
            if self.pneumo is not None:
                if self.pneumo == 'only positive':
                    slice_index = self.df['Pneumonia'] == 1
                    self.df = self.df[slice_index]
                    self.meta_df = self.meta_df[slice_index]
                elif self.pneumo == 'no positive':
                    slice_index = self.df['Pneumonia'] != 1
                    self.df = self.df[slice_index]
                    self.meta_df = self.meta_df[slice_index]
            traindf, valdf, trainmetadf, valmetadf = grouped_split(
                    self.df,
                    self.meta_df,
                    random_state=random_state,
                    test_size=0.05)
            if self.fold == 'train':
                self.df = traindf
                self.meta_df = trainmetadf
            else:
                self.df = valdf
                self.meta_df = valmetadf
        elif self.fold == 'test':
            # Follow the same procedure as above to select the test fold
            foldpath = os.path.join(
                    self.path_to_images,
                    '../labels/test_list.txt')
            folddf = pandas.read_csv(foldpath, names=["Image Index"])
            foldlist = folddf["Image Index"].values.tolist()
            self.df = self.df.loc[foldlist, :]
            self.meta_df = self.meta_df.loc[foldlist, :]
            if subsample_test:
                random.seed(random_state)
                selected_pats = random.sample(list(self.df.index),50)
                self.df = self.df.loc[selected_pats,:]
                self.meta_df = self.meta_df.loc[selected_pats,:]
        else:
            raise ValueError("Invalid fold: {:s}".format(str(self.fold)))
            
        if labels.lower() == 'chestx-ray14':
            self.labels = [
                'Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Effusion',
                'Emphysema',
                'Fibrosis',
                'Hernia',
                'Infiltration',
                'Mass',
                'Nodule',
                'Pleural_Thickening',
                'Pneumonia',
                'Pneumothorax',
                'COVID']

        elif labels.lower() == 'chexpert':
            self.labels = [
                'N/A',
                'Cardiomegaly',
                'N/A',
                'N/A',
                'Edema',
                'Consolidation',
                'Pneumonia',
                'Atelectasis',
                'Pneumothorax',
                'Effusion',
                'N/A',
                'N/A',
                'N/A',
                'COVID']
        else:
            raise ValueError('Invalid value of keyword argument "labels": {:s}.'
                             .format(labels) +\
                             ' Must be one of "CheXpert" or "ChestX-ray14"')
        
