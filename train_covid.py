#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#

import os
import sys

import argparse
import numpy
import pandas
import sklearn.metrics

from models import CXRClassifier
from datasets import ChestXray14H5Dataset, PadChestH5Dataset
from datasets import GitHubCOVIDDataset, BIMCVCOVIDDataset
from datasets import BIMCVNegativeDataset
from datasets import DomainConfoundedDataset

def load_overlap(path="data/bimcv-/listjoin_ok.tsv"):
    neg_overlap_map = {}
    pos_overlap_map = {}
    with open(path, 'r') as handle:
        handle.readline()
        for line in handle:
            idx, neg_id, pos_id = line.split()
            neg_overlap_map[neg_id] = idx
            pos_overlap_map[pos_id] = idx
    return neg_overlap_map, pos_overlap_map

def ds3_grouped_split(df1, df2, random_state=None, test_size=0.05):
    '''
    Split a dataframe such that patients are disjoint in the resulting folds.
    The dataframe must have an index that contains strings that may be processed
    by ds3_get_patient_id to return the unique patient identifiers.
    '''
    neg_overlap_map, pos_overlap_map = load_overlap()
    groups = ds3_get_unique_patient_ids(df1, df2, neg_overlap_map, pos_overlap_map)
    traingroups, testgroups = sklearn.model_selection.train_test_split(
            groups,
            random_state=random_state,
            test_size=test_size)
    traingroups = set(traingroups)
    testgroups = set(testgroups)

    traindict1 = {}
    testdict1 = {}
    traindict2 = {}
    testdict2 = {}
    for idx, row in df1.iterrows():
        patient_id = ds3_get_patient_id(df1, idx, neg_overlap_map)
        if patient_id in traingroups:
            traindict1[idx] = row
        elif patient_id in testgroups:
            testdict1[idx] = row
    for idx, row in df2.iterrows():
        patient_id = ds3_get_patient_id(df2, idx, pos_overlap_map)
        if patient_id in traingroups:
            traindict2[idx] = row
        elif patient_id in testgroups:
            testdict2[idx] = row
    traindf1 = pandas.DataFrame.from_dict(
        traindict1, 
        orient='index',
        columns=df1.columns)
    testdf1 = pandas.DataFrame.from_dict(
        testdict1, 
        orient='index',
        columns=df1.columns)
    traindf2 = pandas.DataFrame.from_dict(
        traindict2, 
        orient='index',
        columns=df2.columns)
    testdf2 = pandas.DataFrame.from_dict(
        testdict2, 
        orient='index',
        columns=df2.columns)
    return traindf1, testdf1, traindf2, testdf2

def ds3_get_patient_id(df, idx, jointlist):
    participant_id = df['participant'].loc[idx]
    try:
        val = jointlist[participant_id]
        print(val)
        return val
    except KeyError:
        return participant_id

def ds3_get_unique_patient_ids(df1, df2, neg_overlap_map, pos_overlap_map):
    # check that ids don't overlap to start
    if len(set(df1.participant).intersection(set(df2.participant))) != 0:
        print(df1.participant[:4])
        print(df2.participant[:4])
        #print(set(df1.participant).intersection(set(df2.participant)))
        raise ValueError
    neg_idxs = [ds3_get_patient_id(df1, idx, neg_overlap_map) for idx in df1.index]
    pos_idxs = [ds3_get_patient_id(df2, idx, pos_overlap_map) for idx in df2.index]
    neg_idxs = list(set(neg_idxs))
    pos_idxs = list(set(pos_idxs))
    neg_idxs.sort()
    pos_idxs.sort()
    return neg_idxs + pos_idxs

def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label.lower():
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))

def train_dataset_1(seed, alexnet=False, freeze_features=False):
    trainds = DomainConfoundedDataset(
            ChestXray14H5Dataset(fold='train', labels='chestx-ray14', random_state=seed),
            GitHubCOVIDDataset(fold='train', labels='chestx-ray14', random_state=seed)
            )

    valds = DomainConfoundedDataset(
            ChestXray14H5Dataset(fold='val', labels='chestx-ray14', random_state=seed),
            GitHubCOVIDDataset(fold='val', labels='chestx-ray14', random_state=seed)
            )

    # generate log and checkpoint paths
    if alexnet: netstring = 'alexnet'
    elif freeze_features: netstring = 'densenet121frozen'
    else: netstring = 'densenet121'
    logpath = 'logs/dataset1.{:s}.{:d}.log'.format(netstring, seed)
    checkpointpath = 'checkpoints/dataset1.{:s}.{:d}.pkl'.format(netstring, seed)

    classifier = CXRClassifier()
    classifier.train(trainds,
                valds,
                max_epochs=30,
                lr=0.01, 
                weight_decay=1e-4,
                logpath=logpath,
                checkpoint_path=checkpointpath,
                verbose=True,
                scratch_train=alexnet,
                freeze_features=freeze_features)

def train_dataset_2(seed, alexnet=False, freeze_features=False):
    trainds = DomainConfoundedDataset(
            PadChestH5Dataset(fold='train', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='train', labels='chestx-ray14', random_state=seed)
            )
    valds = DomainConfoundedDataset(
            PadChestH5Dataset(fold='val', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='val', labels='chestx-ray14', random_state=seed)
            )

    # generate log and checkpoint paths
    if alexnet: netstring = 'alexnet'
    elif freeze_features: netstring = 'densenet121frozen'
    else: netstring = 'densenet121'
    logpath = 'logs/dataset2.{:s}.{:d}.log'.format(netstring, seed)
    checkpointpath = 'checkpoints/dataset2.{:s}.{:d}.pkl'.format(netstring, seed)

    classifier = CXRClassifier()
    classifier.train(trainds,
                valds,
                max_epochs=30,
                lr=0.01, 
                weight_decay=1e-4,
                logpath=logpath,
                checkpoint_path=checkpointpath,
                verbose=True,
                scratch_train=alexnet,
                freeze_features=freeze_features)

def train_dataset_3(seed, alexnet=False, freeze_features=False):
    # Unlike the other datasets, there is overlap in patients between the
    # BIMCV-COVID-19+ and BIMCV-COVID-19- datasets, so we have to perform the 
    # train/val/test split *after* creating the datasets.

    # Start by getting the *full* dataset - not split!
    trainds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', random_state=seed)
            )
    valds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', random_state=seed)
            )
    # split on a per-patient basis
    trainvaldf1, testdf1, trainvaldf2, testdf2 = ds3_grouped_split(trainds.ds1.df, trainds.ds2.df, random_state=seed)
    traindf1, valdf1, traindf2, valdf2 = ds3_grouped_split(trainvaldf1, trainvaldf2, random_state=seed)

    # Update the dataframes to respect the per-patient splits
    trainds.ds1.df = traindf1
    trainds.ds2.df = traindf2
    valds.ds1.df = valdf1
    valds.ds2.df = valdf2
    trainds.len1 = len(trainds.ds1)
    trainds.len2 = len(trainds.ds2)
    valds.len1 = len(valds.ds1)
    valds.len2 = len(valds.ds2)

    # generate log and checkpoint paths
    if alexnet: netstring = 'alexnet'
    elif freeze_features: netstring = 'densenet121frozen'
    else: netstring = 'densenet121'
    logpath = 'logs/dataset3.{:s}.{:d}.log'.format(netstring, seed)
    checkpointpath = 'checkpoints/dataset3.{:s}.{:d}.pkl'.format(netstring, seed)

    classifier = CXRClassifier()
    classifier.train(trainds,
                valds,
                max_epochs=30,
                lr=0.01, 
                weight_decay=1e-4,
                logpath=logpath,
                checkpoint_path=checkpointpath,
                verbose=True,
                scratch_train=alexnet,
                freeze_features=freeze_features)

def main():
    parser = argparse.ArgumentParser(description='Training script for COVID-19 '
            'classifiers. Make sure that datasets have been set up before '
            'running this script. See the README file for more information.')
    parser.add_argument('--dataset', dest='dataset', type=int, default=1, required=False,
                        help='The dataset number on which to train. Choose "1" or "2" or "3".')
    parser.add_argument('--seed', dest='seed', type=int, default=30493, required=False,
                        help='The random seed used to generate train/val/test splits')
    parser.add_argument('--network', dest='network', type=str, default='densenet121', required=False,
                        help='The network type. Choose "densenet121", "logistic", or "alexnet".')
    parser.add_argument('--device-index', dest='deviceidx', type=int, default=None, required=False,
                        help='The index of the GPU device to use. If None, use the default GPU.')
    args = parser.parse_args()

    for dirname in ['checkpoints', 'logs']:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

    if args.deviceidx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{:d}".format(args.deviceidx)

    if args.dataset == 1:
        train_dataset_1(args.seed, 
                alexnet=(args.network.lower() == 'alexnet'), 
                freeze_features=(args.network.lower() == 'logistic'))
    if args.dataset == 2:
        train_dataset_2(args.seed, 
                alexnet=(args.network.lower() == 'alexnet'), 
                freeze_features=(args.network.lower() == 'logistic'))
    if args.dataset == 3:
        train_dataset_3(args.seed, 
                alexnet=(args.network.lower() == 'alexnet'), 
                freeze_features=(args.network.lower() == 'logistic'))

if __name__ == "__main__":
    main()
