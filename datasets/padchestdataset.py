#!/usr/bin/env python
import os

import numpy
import pandas
import sklearn.model_selection
from PIL import Image
import random
from torchvision import transforms

from datasets.cxrdataset import CXRDataset
import datasets.padchestmap as padchestmap

NUMPY_IMAGE_BITS = 8
CORRUPTED=['216840111366964013686042548532013208193054515_02-026-007.png',
           '216840111366964013590140476722013058110301622_02-056-111.png',
           '216840111366964013649110343042013092101343018_02-075-146.png',
           '216840111366964013590140476722013043111952381_02-065-198.png',
           '216840111366964013590140476722013049100117076_02-063-097.png',
           '216840111366964013590140476722013028161046120_02-015-149.png',
           '216840111366964013829543166512013353113303615_02-092-190.png',
           '216840111366964013962490064942014134093945580_01-178-104.png',
           '216840111366964012989926673512011151082430686_00-157-045.png',
           '216840111366964012989926673512011132200139442_00-157-099.png',
           '216840111366964013076187734852011291090445391_00-196-188.png',
           '216840111366964012373310883942009117084022290_00-064-025.png',
           '216840111366964012989926673512011101154138555_00-191-086.png',
           '216840111366964012339356563862009072111404053_00-043-192.png',
           '216840111366964012558082906712009301143450268_00-075-157.png',
           '216840111366964012487858717522009280135853083_00-075-001.png',
           '216840111366964012283393834152009033140208626_00-059-118.png',
           '216840111366964012283393834152009033102258826_00-059-087.png',
           '216840111366964012373310883942009170084120009_00-097-074.png',
           '216840111366964012373310883942009180082307973_00-097-011.png',
           '216840111366964012819207061112010281134410801_00-129-131.png',
           '216840111366964012339356563862009068084200743_00-045-105.png',
           '216840111366964012558082906712009300162151055_00-078-079.png',
           '216840111366964012989926673512011074122523403_00-163-058.png',
           '216840111366964012558082906712009327122220177_00-102-064.png',
           '216840111366964012373310883942009152114636712_00-102-045.png',
           '216840111366964012989926673512011083134050913_00-168-009.png',
           '216840111366964012959786098432011033083840143_00-176-115.png',
           '216840111366964013076187734852011178154626671_00-145-086.png',
           '216840111366964013076187734852011287092959219_00-195-171.png',
           '216840111366964012819207061112010306085429121_04-020-102.png',
           '216840111366964012819207061112010315104455352_04-024-184.png',
           '216840111366964012819207061112010307142602253_04-014-084.png']

def grouped_split(dataframe, random_state=None, test_size=0.05):
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
        patient_id = _get_patient_id(dataframe, idx)
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
    return traindf, testdf

def _get_patient_id(df, idx):
    return df['PatientID'].loc[idx]

def _get_unique_patient_ids(dataframe):
    ids = list(dataframe.index)
    ids = [_get_patient_id(dataframe, i) for i in ids]
    ids = list(set(ids))
    ids.sort()
    return ids

class PadChestDataset(CXRDataset):
    def __init__(
            self,
            fold,
            include_lateral=False,
            random_state=30493,
            labels='CheXpert',
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
            include_lateral (bool): If True, include the lateral radiograph
                views in the dataset. If False, include only frontal views.
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
        self.fold = fold
        self.labelstyle = labels.lower()
        self.transform = self._transforms[fold]
        self.path_to_images = "data/PadChest/"
        self.path_to_labels = os.path.join(self.path_to_images, 
                "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
        self.path_to_images = os.path.join(self.path_to_images, 'images')
        self.pneumo = pneumo

        # Load files containing labels, and perform train/valid split if necessary
        self.df = pandas.read_csv(self.path_to_labels)
        # Blacklist corrupted files
        for fname in CORRUPTED:
            self.df.drop(self.df[self.df['ImageID'] == fname].index, inplace=True)
        if not include_lateral:
            self.df = self.df.query('''Projection == 'PA' or Projection == 'AP' ''')

        trainvaldf, testdf = grouped_split(
                self.df, 
                random_state=random_state,
                test_size=0.05)

        traindf, valdf = grouped_split(
                trainvaldf,
                random_state=random_state,
                test_size=0.05)
        if self.fold == 'train':
            self.df = traindf
            
            if self.pneumo is not None:
                if self.pneumo == 'only positive':
                    self.df = self.df[self.df['Pneumonia'] == 1]
                elif self.pneumo == 'no positive':
                    self.df = self.df[self.df['Pneumonia'] != 1]
                    
        elif self.fold == 'val':
            self.df = valdf
            
            if self.pneumo is not None:
                if self.pneumo == 'only positive':
                    self.df = self.df[self.df['Pneumonia'] == 1]
                elif self.pneumo == 'no positive':
                    self.df = self.df[self.df['Pneumonia'] != 1]
                    
        elif self.fold == 'test':
            self.df = testdf
            if subsample_test:
                random.seed(random_state)
                selected_pats = random.sample(list(self.df.index),100)
                self.df = self.df.loc[selected_pats,:]
        else:
            raise ValueError("Invalid fold: {:s}".format(str(fold)))
            
        if self.labelstyle == 'chestx-ray14':
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

        elif self.labelstyle == 'chexpert':
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

    def __getitem__(self, idx):
        image = self._raw_image_from_disk(idx)
        # manually convert to 8-bit image. This happens with the RGB 
        # conversion anyway, but PIL has a weird thresholding behavior with 
        # direct RGB conversion.
        image = numpy.array(image)
        image = image/(2**8)
        image = image.astype(numpy.uint8)
        image = Image.fromarray(image, mode='L')

        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = self.get_labels(idx)

        return (image, labels, 0, 0)

    def _raw_image_from_disk(self, idx):
        '''
        Retrieve the raw PIL image from storage.
        '''
        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.ImageID.iloc[idx]))
        return image

    def _parse_labels(self, imageid):
        '''
        This requires a few steps:
        1) The Labels field of the csv file contains a string, which can
           possibly be evaluated to a Python list, or `NaN`. 
        2) Handle the label `unchanged`. Look to the most recent previous study
           and copy over the labels. This may have to be done recursively, and 
           sometimes the oldest study may not contain labels. In this case, we 
           give the study no labels.
        '''
        # s will be something like 
        #'["costophrenic angle blunting", "loculated pleural effusion", "clavicle fracture"]'
        row = self.df.query('''ImageID == "{:s}" '''.format(imageid))
        idx = row.index[0]
        s = row.Labels.iloc[0]
        # convert the string to a list
        if isinstance(s, str):
            findings = eval(s)
        else:
            findings = ''
        # strip white space
        findings = [f.strip() for f in findings]
        
        if 'unchanged' in findings:
            # Look backward toward the most recent study
            patientid = self.df.PatientID[idx]
            studydate = self.df.StudyDate_DICOM[idx] # numpy.int64
            other_studies = self.df.query(''' PatientID == "{:s}" '''.format(patientid))
            # Luckily, no studies with the 'unchanged' label have other studies
            # from the same day, so we can always look to previous days to find
            # previous studies
            previous_studies = other_studies.query("StudyDate_DICOM < {:d}".format(studydate))
            if len(previous_studies) > 0:
                most_recent_study = other_studies.sort_values(by="StudyDate_DICOM").iloc[0]
                findings.remove("unchanged")
                findings += self._parse_labels(most_recent_study.ImageID) 
            else:
                findings.remove("unchanged")
        return findings
        

    def get_labels(self, idx):
        '''
        Get the labels for index ``idx``.

        An initial draft of the labels is obtained via _parse_labels, which is 
        then converted to the desired style of labels.
        '''
        imageid = self.df.ImageID.iloc[idx]
        findings = self._parse_labels(imageid)

        if self.labelstyle == 'chexpert':
            map_ = padchestmap.padchesttochexpert
        elif self.labelstyle == 'chestx-ray14':
            map_ = padchestmap.padchesttochestxray14
        else:
            raise NotImplementedError

        labels = []
        for f in findings:
            # map the PadChest finding to the CheXpert label
            if f == '':
               continue
            converted = map_[f]
            if converted == '':
               continue 
            elif isinstance(converted, str):
                labels.append(converted)
            elif isinstance(converted, list):
                labels += converted
            else:
                raise ValueError(str(converted))
        labels = set(labels)

        # convert to one-hot encoding
        label = numpy.zeros(len(self.labels), dtype=int)
        for i in range(0, len(self.labels)):
            if self.labels[i] != "N/A" and self.labels[i] in labels:
                label[i] = 1
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
            output[isample] = self.get_labels(isample)
        return output
