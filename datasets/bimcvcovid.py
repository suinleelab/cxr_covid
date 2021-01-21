#!/usr/bin/env python
import yaml
import h5py
import io
import numpy
import os
import pandas
import re
import sklearn.model_selection
import torch
import datasets.padchestmap as padchestmap
from datasets.cxrdataset import CXRDataset, H5Dataset
from PIL import Image

padchesttochexpert = padchestmap.padchesttochexpert
padchesttochexpert['COVID 19'] = 'COVID-19'
padchesttochexpert['COVID 19 uncertain'] = 'COVID-19'
padchesttochexpert['viral pneumonia'] = 'Pneumonia'

padchesttochestxray14 = padchestmap.padchesttochestxray14
padchesttochestxray14['COVID 19'] = 'COVID-19'
padchesttochestxray14['COVID 19 uncertain'] = 'COVID-19'
padchesttochestxray14['viral pneumonia'] = 'Pneumonia'

SKIP_WINDOWING = [
        "sub-S03169_ses-E07193_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03169_ses-E07878_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03067_ses-E07350_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03067_ses-E07908_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03240_ses-E07784_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03218_ses-E07970_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03240_ses-E07626_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03214_ses-E07979_run-1_bp-chest_vp-ap_dx.png"
        ]

FLIP = ["sub-S03936_ses-E08636_run-1_bp-chest_vp-ap_dx.png",
        "sub-S04275_ses-E08736_run-1_bp-chest_vp-ap_dx.png",
        "sub-S04190_ses-E08730_run-1_bp-chest_vp-ap_dx.png",
        "sub-S03068_ses-E06592_run-1_bp-chest_vp-ap_cr.png"]

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
    return df['participant'].loc[idx]

def _get_unique_patient_ids(dataframe):
    ids = list(dataframe.index)
    ids = [_get_patient_id(dataframe, i) for i in ids]
    ids = list(set(ids))
    ids.sort()
    return ids


class BIMCVCOVIDDataset(H5Dataset, CXRDataset):
    '''
    projection labels include AP, PA, LAT, and AP_SUPINE (based on DICOM data).
      if 'include_unknown_projections == True' then the additional hand-labeled 
      projections 'frontal' and 'lateral' will be included. 
    '''
    def __init__(self, fold, random_state=30493, include_lateral=False, 
                 include_unknown_projections=False, include_ap_supine=False,
                 include_unknown_labels=False, initialize_h5=False, covid_labels='molecular',
                 labels='chexpert', projection=None):
        '''
        covid_labels (str): if 'molecular', the COVID label associated with each
          image will be based on molecular assay results (PCR or serology); all 
          images will be COVID positive. If 'radiologic', the COVID label will
          be based on presence of radiological evidence of COVID.
        '''
        self.labelstyle = labels.lower()
        if self.labelstyle == 'chexpert':
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
                'COVID-19']
        elif self.labelstyle == 'chestx-ray14':
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
                'COVID-19']
        else:
            raise NotImplementedError
        self.covid_labels = covid_labels.lower()
        if not self.covid_labels in ['radiologic', 'molecular']:
            raise ValueError('Invalid value {:s} for keyword argument "covid_labels"'.format(repr(self.covid_labels)))
        if self.covid_labels == 'radiologic':
            padchesttochestxray14['viral pneumonia'] = 'COVID-19'
            padchesttochexpert['viral pneumonia'] = 'COVID-19'
        self.report_regex = re.compile('ses-E\d+')
        self._set_datapaths()
        # Filter images with windowing data
        self.df = self.df.query('window_center == window_center | lut == lut') # remove NaN
        self.df.lut = self.df.lut.apply(lambda x: eval(x) if isinstance(x,str) else x) # strings to LUT lists
        if include_unknown_projections: # unlabeled projections
            with open(self.manual_projection_label_path, 'r') as handle:
                projection_labels = yaml.load(handle)
            unknowns = self.df.query("projection == 'UNK'")
            for idx, row in unknowns.iterrows():
                self.df.projection.loc[idx] = projection_labels[row.path]
        else:
            self.df = self.df.query("projection == 'AP' or projection == 'PA' or projection == 'LAT' or projection == 'AP SUPINE'")
        if not include_lateral: # lateral projections
            self.df = self.df.query("projection == 'AP' or projection == 'PA' or projection == 'frontal' or projection == 'AP SUPINE'")
        if not include_ap_supine:
            self.df = self.df.query("projection == 'AP' or projection == 'PA' or projection == 'frontal' or projection == 'LAT'")
        if not include_unknown_labels:
            with open(self.unknown_label_path, 'r') as handle:
                unknown_paths = []
                for line in handle:
                    unknown_paths.append(line.strip())
                self.df = self.df[~self.df['path'].isin(unknown_paths)]
        if not fold in ['train', 'val', 'test', 'all']:
            raise ValueError("Invalid fold: {:s}".format(fold))
        self._transforms['all'] = self._transforms['val']
        self.transform = self._transforms[fold]
        if fold == 'all':
            pass
        else:
            trainvaldf, testdf = grouped_split(
                    self.df, 
                    random_state=random_state,
                    test_size=0.05)
            if fold == 'train' or fold == 'val':
                traindf, valdf = grouped_split(
                        trainvaldf,
                        random_state=random_state,
                        test_size=0.05)
                if fold == 'train':
                    self.df = traindf
                    if projection is not None:
                        if projection == 'AP':
                            self.df = self.df[self.df.projection == 'AP']
                        elif projection == 'PA':
                            self.df = self.df[self.df.projection == 'PA']
                elif fold == 'val':
                    self.df = valdf
                    if projection is not None:
                        if projection == 'AP':
                            self.df = self.df[self.df.projection == 'AP']
                        elif projection == 'PA':
                            self.df = self.df[self.df.projection == 'PA']
            elif fold == 'test':
                self.df = testdf
                if projection is not None:
                    if projection == 'AP':
                        self.df = self.df[self.df.projection == 'AP']
                    elif projection == 'PA':
                        self.df = self.df[self.df.projection == 'PA']
        if initialize_h5:
            self.init_worker(None)

    def _set_datapaths(self):
        self.datapath = 'data/bimcv+'
        self.labelpath = 'derivatives/labels/labels_covid19_posi.tsv'
        self.unknown_label_path = 'datasets/bimcv_covid_unknown_labels.txt'
        self.labeldf = pandas.read_csv(os.path.join(self.datapath, self.labelpath), delimiter='\t')
        self.h5path = 'data/bimcv+/bimcv+.h5'
        self.df = pandas.read_csv(os.path.join(self.datapath, 'bimcv+.csv'))
        self.manual_projection_label_path = "datasets/bimcv_covid_manual_projection_labels.yml"

    def init_worker(self, worker_id):
        self.h5 = h5py.File(self.h5path, 'r', swmr=True)

    def __getitem__(self, idx):
        UINT8_MAX = 255 

        image = self._raw_image_from_disk(idx)
        imagename = self.df.path.iloc[idx].split('/')[-1]
        if imagename in SKIP_WINDOWING:
            pass
        else:
            image = numpy.array(image, dtype=numpy.int64)
            # Use LUT if we have it 
            lut = self.df.lut.iloc[idx]
            if isinstance(lut, list):
                lut_min = int(self.df.lut_min.iloc[idx])
                lut = numpy.array(lut)
                lut = numpy.concatenate((numpy.ones(lut_min)*lut[0],
                                         lut,
                                         numpy.ones(65536-lut_min-len(lut))*lut[-1]), axis=0)
                image = lut[image]
                if self.df.rescale_slope.iloc[idx]:
                    image *= self.df.rescale_slope.iloc[idx] + self.df.rescale_intercept.iloc[idx]
                max_ = 2**self.df.bits_stored.iloc[idx]-1
                image = image.astype(numpy.float64)/max_
            else: # use window data
                window_center = self.df.window_center.iloc[idx]
                window_width = self.df.window_width.iloc[idx]
                window_min = int(window_center - window_width/2)
                image -= window_min
                image = image.astype(numpy.float64)*1/window_width
            # clip
            image[image<0] = 0
            image[image>1] = 1
            image = (image*UINT8_MAX).astype(numpy.uint8)

            # invert if needed
            photometric_interpretation = self.df.photometric_interpretation.iloc[idx]
            if photometric_interpretation == 'MONOCHROME1':
                image = UINT8_MAX-image
            elif photometric_interpretation == 'MONOCHROME2':
                image = image
            else:
                raise ValueError('unknown photometric interpretation: {:s}'.format(photometric_interpretation))
            if imagename in FLIP:
                image = numpy.flipud(image)
            try:
                image = Image.fromarray(image, mode='L')
            except ValueError:
                image = Image.fromarray(image, mode='RGB')
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = self.get_labels(idx)
        return image, label, 0, 0 

    def _raw_image_from_disk(self, idx):
        '''
        Retrieve the raw PIL image from storage.
        '''
        imagename = self.df.path.iloc[idx].split('/',1)[1]
        data = self.h5['images'].get(imagename)
        image = Image.open(io.BytesIO(numpy.array(data)))
        return image

    def add_covid_label(self, findings):
        # ALL samples are covid-19+
        if not 'COVID 19' in findings:
            findings.append('COVID 19')
        return findings

    def get_labels(self, idx):
        '''
        Get the labels for index ``idx``.

        An initial draft of the labels is obtained via _parse_labels, which is 
        then converted to the desired style of labels.
        '''
        findings = self._parse_labels(idx)
        if self.covid_labels == 'molecular':
            findings = self.add_covid_label(findings)
        if self.labelstyle == 'chexpert':
            map_ = padchesttochexpert
        elif self.labelstyle == 'chestx-ray14':
            map_ = padchesttochestxray14
        else:
            raise ValueError
        labels = []
        for f in findings:
            # map the PadChest finding to the CheXpert/chestx-ray14 label
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

    def _parse_labels(self, idx):
        '''
        This requires a few steps:
        1) The Labels field of the tsv file contains a string, which can
           possibly be evaluated to a Python list, or `NaN`. 
        2) Handle the label `unchanged`. Look to the most recent previous study
           and copy over the labels. This may have to be done recursively, and 
           sometimes the oldest study may not contain labels. In this case, we 
           give the study no labels.
        '''
        path = self.df.path.iloc[idx]
        reportid = self.report_regex.search(path).group()
        # s will be something like 
        #'["costophrenic angle blunting", "loculated pleural effusion", "clavicle fracture"]'
        row = self.labeldf.query("ReportID == @reportid")
        try:
            labeldf_idx = row.index[0]
        except IndexError: # no report found!
            #with open('temp.txt', 'a') as handle:
            #    handle.write(path + '\n')
            return []
        s = row.Labels.iloc[0]
        # convert the string to a list
        if isinstance(s, str):
            findings = [substr.strip().strip("'").strip() for substr in s.strip("[]").split('\t')]
        else:
            findings = []

        if 'unchanged' in findings:
            # Look backward toward the most recent study
            patientid = self.labeldf.PatientID[labeldf_idx]
            study_date = self.df.study_date.iloc[idx] # numpy.int64
            study_time = self.df.study_time.iloc[idx]
            other_studies = self.df.query(''' participant == "{:s}" '''.format(patientid))
            previous_studies = other_studies.query("study_date < @study_date | (study_date == @study_date & study_time < @study_time)")
            if len(previous_studies) > 0:
                other_studies = previous_studies.sort_values(by="study_time", kind='mergesort') 
                # stable sort here to previous time ordering
                most_recent_study = previous_studies.sort_values(by="study_date", kind='mergesort').iloc[-1]
                findings.remove("unchanged")
                new_idx = self.df.index.get_loc(most_recent_study.name)
                findings += self._parse_labels(new_idx) 
            else:
                findings.remove("unchanged")
        return findings

    def get_all_labels(self):
        arr = numpy.zeros((len(self), len(self.labels)))
        for i in range(len(self)):
            arr[i] = self.get_labels(i)
        return arr
