#!/usr/bin/env python
import os

import numpy
import pandas
import sklearn.model_selection
from PIL import Image

from datasets.cxrdataset import CXRDataset

def grouped_split(dataframe, random_state, test_size=0.05):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by _get_patient_id to return the unique patient identifiers.
    '''
    groups = list(set(dataframe['Patient ID']))
    groups.sort()
    traingroups, testgroups = sklearn.model_selection.train_test_split(
            groups,
            random_state=random_state,
            test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    trainidx = []
    testidx = []
    for idx, row in dataframe.iterrows():
        patient_id = row['Patient ID']
        if patient_id in traingroups:
            trainidx.append(idx)
        elif patient_id in testgroups:
            testidx.append(idx)
    traindf = dataframe.loc[dataframe.index.isin(trainidx),:]
    testdf = dataframe.loc[dataframe.index.isin(testidx),:]
    return traindf, testdf


def _convert_dataframe(df):
    '''
    Convert the labels in 'metadata.csv' to one-hot encoded labels and
    return a new dataframe.
    '''
    ## filter to just include x-rays (CTs present in this dataset)
    df = df[df.modality == 'X-ray']
    
    ## filter lateral images
    df = df[df.view != 'L']
    
    ## filter supine
    df = df[df.view != 'AP Supine']
    
    ## set columns to chexpert labels + COVID
    cols = ['Enlarged Cardiomediastinum',
                'Cardiomegaly',
                'Lung Opacity',
                'Lung Lesion',
                'Edema',
                'Consolidation',
                'Pneumonia',
                'Atelectasis',
                'Pneumothorax',
                'Pleural Effusion',
                'Pleural Other',
                'Fracture',
                'Support Devices',
                'COVID',
                'Patient ID']
    
    ## maps sets of possible findings to labels
    covid_set = ['COVID-19','COVID-19, ARDS']
    pneumo_set = ['ARDS',
                 'Bacterial',
                 'Chlamydophila',
                 'E.Coli',
                 'Influenza',
                 'Klebsiella',
                 'Legionella',
                 'Lipoid',
                 'Mycoplasma Bacterial Pneumonia',
                 'Pneumocystis',
                 'Pneumonia',
                 'SARS',
                 'Streptococcus',
                 'Varicella']
    healthy_set = ['No Finding']
    
    # make new DF to hold labels
    new_data = numpy.zeros((df.shape[0],len(cols)))
    new_df = pandas.DataFrame(data=new_data,columns=cols,index=df.filename)
    
    for i, (irow, row) in enumerate(df.iterrows()):
        
        if row['finding'] in covid_set:
            new_df.iloc[i]['COVID'] = 1
        elif row['finding'] in pneumo_set:
            new_df.iloc[i]['Pneumonia'] = 1
        
        if row['intubation_present'] == 'Y':
            new_df.iloc[i]['Support Devices'] = 1
    
    new_df['Patient ID'] = df.patientid.values
    new_df['Projection'] = df.view.values
    new_df['Sex'] = df.sex.values

    return new_df

class GitHubCOVIDDataset(CXRDataset):
    def __init__(
            self,
            fold,
            random_state=30493,
            labels='CheXpert'):
        '''
        Create a dataset of the COVID images for use in a PyTorch model.

        Args:
            fold (str): The shard of the COVID data that the dataset should
                contain. One of either 'train', 'val', or 'test'.
            random_state (int): An integer used to see generation of the 
                train/val/test split from the patients specified in the 
                'metadata.csv' file provided with the COVID dataset. 
                Used to ensure reproducability across runs.
            labels (str): One of either 'CheXpert' or 'ChestX-ray14'. In either
                case, each label will be a boolean array where each element of 
                the array corresponds to a pathology, 1 indicates a 'positive 
                mention' of the pathology, and 0 indicates any of 'at least one 
                uncertain mention with no positive mentions', 'a negative 
                mention', or 'no mention'. If 'CheXpert', the labels will 
                include all pathologies specified by the CheXpert labeler, i.e.,

                    0:  Enlarged Cardiomediastinum
                    1:  Cardiomegaly
                    2:  Lung Opacity
                    3:  Lung Lesion
                    4:  Edema
                    5:  Consolidation
                    6:  Pneumonia
                    7:  Atelectasis
                    8:  Pneumothorax
                    9:  Pleural Effusion
                    10: Pleural Other
                    11: Fracture
                    12: Support Devices               

                If 'ChestX-ray14', the labels will include only pathologies 
                specified in both the 'ChestX-ray14' and 'CheXpert' datasets, 
                i.e.,
                    0:  Atelectasis
                    1:  Cardiomegaly
                    2:  Pleural Effusion ('Effusion' in ChestX-ray14)
                    3:  N/A
                    4:  N/A
                    5:  N/A
                    6:  Pneumonia
                    7:  Pneumothorax
                    8:  Consolidation
                    9:  Edema
                    10: N/A
                    11: N/A
                    12: N/A
                    13: N/A

                where presence of 'N/A' labels and the order of the pathologies
                is chosen for compatibility with classifiers trained on the 
                ChestX-ray14 data.
        '''

        self.transform = self._transforms[fold]
        self.path_to_images = "data/covid-chestxray-dataset/images/"
        self.has_appa = False

        # Load files containing labels, and perform train/valid split if necessary
        metadatapath = "data/covid-chestxray-dataset/metadata.csv"
        self.df = pandas.read_csv(metadatapath)
        self.df = _convert_dataframe(self.df)
        
        train, test = grouped_split(
                    self.df,
                    random_state=random_state,
                    test_size=0.10)
        
        train, val = grouped_split(
                    train,
                    random_state=random_state,
                    test_size=0.10)
        
        if fold == 'train':
            self.df = train
        elif fold == 'val':
            self.df = val
        elif fold == 'test':
            self.df = test
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))
        
        if labels.lower() == 'chestx-ray14':
            self.labels = [
                'Atelectasis',
                'Cardiomegaly',
                'Consolidation',
                'Edema',
                'Pleural Effusion',
                'N/A',
                'N/A',
                'N/A',
                'N/A',
                'N/A',
                'N/A',
                'N/A',
                'Pneumonia',
                'Pneumothorax',
                'COVID']
        elif labels.lower() == 'chexpert':
            self.labels = [
                'Enlarged Cardiomediastinum',
                'Cardiomegaly',
                'Lung Opacity',
                'Lung Lesion',
                'Edema',
                'Consolidation',
                'Pneumonia',
                'Atelectasis',
                'Pneumothorax',
                'Pleural Effusion',
                'Pleural Other',
                'Fracture',
                'Support Devices',
                'COVID']
        else:
            raise ValueError('Invalid value of keyword argument "labels": {:s}.'
                             .format(labels) +\
                             ' Must be one of "CheXpert" or "ChestX-ray14"')
