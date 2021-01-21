#!/usr/bin/env python
# roc_bimcv.py
#
# Script to calculate ROC curves and AUROC values for dataset III; external 
# tests are performed against dataset I.
#
import numpy
import sklearn.metrics

from models import CXRClassifier
from datasets import ChestXray14H5Dataset, PadChestH5Dataset
from datasets import GitHubCOVIDDataset, BIMCVCOVIDDataset
from datasets import BIMCVNegativeDataset
from datasets import DomainConfoundedDataset
from train_covid import _find_index, ds3_get_patient_id, ds3_get_unique_patient_ids, ds3_grouped_split, load_overlap

import matplotlib
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
import matplotlib.pyplot as pyplot

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

#################################### Options ##################################
# Modify to match the output files from your training procedure. The random
# seeds are automatically parsed from the file names, and must be the same for
# the two datasets!

# Models trained on dataset I
ds1_checkpoints = \
        ['checkpoints/dataset1.densenet121.30493.pkl.best_auroc',
         'checkpoints/dataset1.densenet121.30494.pkl.best_auroc',
         'checkpoints/dataset1.densenet121.30495.pkl.best_auroc',
         'checkpoints/dataset1.densenet121.30496.pkl.best_auroc',
         'checkpoints/dataset1.densenet121.30497.pkl.best_auroc']

# Models trained on dataset III
ds3_checkpoints = \
        ['checkpoints/dataset3.densenet121.30493.pkl.best_auroc',
         'checkpoints/dataset3.densenet121.30494.pkl.best_auroc',
         'checkpoints/dataset3.densenet121.30495.pkl.best_auroc',
         'checkpoints/dataset3.densenet121.30496.pkl.best_auroc',
         'checkpoints/dataset3.densenet121.30497.pkl.best_auroc']
###############################################################################

def plot(ax, checkpointpath, seed, legend=False):
    githubcxr14_testds = DomainConfoundedDataset(
            ChestXray14H5Dataset(fold='test', labels='chestx-ray14', random_state=seed),
            GitHubCOVIDDataset(fold='test', labels='chestx-ray14', random_state=seed)
            )

    bimcv_testds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='test', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='test', labels='chestx-ray14', random_state=seed)
            )

    # Unlike the other datasets, there is overlap in patients between the
    # BIMCV-COVID-19+ and BIMCV-COVID-19- datasets, so we have to perform the 
    # train/val/test split *after* creating the datasets.

    # Start by getting the *full* dataset - not split!
    bimcv_testds = DomainConfoundedDataset(
            BIMCVNegativeDataset(fold='all', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='all', labels='chestx-ray14', random_state=seed)
            )
    # split on a per-patient basis
    trainvaldf1, testdf1, trainvaldf2, testdf2 = ds3_grouped_split(bimcv_testds.ds1.df, bimcv_testds.ds2.df, random_state=seed)

    # Update the dataframes to respect the per-patient splits
    bimcv_testds.ds1.df = testdf1
    bimcv_testds.ds2.df = testdf2
    bimcv_testds.len1 = len(bimcv_testds.ds1)
    bimcv_testds.len2 = len(bimcv_testds.ds2)

    classifier = CXRClassifier()
    classifier.load_checkpoint(checkpointpath)
    
    githubcxr14_probs = classifier.predict(githubcxr14_testds)
    print(githubcxr14_probs.shape)
    githubcxr14_true = githubcxr14_testds.get_all_labels()
    githubcxr14_idx = _find_index(githubcxr14_testds, 'COVID')

    githubcxr14_auroc = sklearn.metrics.roc_auc_score(
            githubcxr14_true[:, githubcxr14_idx],
            githubcxr14_probs[:, githubcxr14_idx]
            )
    print("githubcxr14 auroc: ", githubcxr14_auroc)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            githubcxr14_true[:, githubcxr14_idx],
            githubcxr14_probs[:, githubcxr14_idx]
            )
    kwargs = {'color': '#b43335', 'linewidth': 1}
    if legend: ax.plot(fpr, tpr, label='ChestX-ray14/\nGitHub-COVID', **kwargs)
    else: ax.plot(fpr, tpr, **kwargs)

    bimcv_probs = classifier.predict(bimcv_testds)
    bimcv_true = bimcv_testds.get_all_labels()
    bimcv_idx = _find_index(bimcv_testds, 'COVID-19')
    bimcv_auroc = sklearn.metrics.roc_auc_score(
            bimcv_true[:, bimcv_idx],
            bimcv_probs[:, bimcv_idx] 
            )
    print("bimcv auroc: ", bimcv_auroc)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            bimcv_true[:, bimcv_idx],
            bimcv_probs[:, bimcv_idx] 
            )
    kwargs = {'color': '#1e579a', 'linewidth': 1}
    if legend: ax.plot(fpr, tpr, label='BIMCV-COVID-19−/\nBIMCV-COVID-19+', **kwargs)
    else: ax.plot(fpr, tpr, **kwargs)
    return githubcxr14_auroc, bimcv_auroc

def main():
    ds1_seeds = [int(os.path.basename(filepath).split('.')[2]) \
                 for filepath in ds1_checkpoints]

    ds3_seeds = [int(os.path.basename(filepath).split('.')[2]) \
                 for filepath in ds3_checkpoints]


    fig = pyplot.gcf()
    fig.set_size_inches(8.8/2.54, 2)
    ax0 = pyplot.subplot(1, 2, 1)
    ax1 = pyplot.subplot(1, 2, 2)
    for ax in [ax0, ax1]:
        ax.plot((0,1), (0,1), color='#a0a0a0', linewidth=1, ls='--')

    ds1_auroc_list = []
    ds3_auroc_list = []
    for i, (path, seed) in enumerate(zip(ds1_checkpoints, ds1_seeds)):
        ds1_auroc, ds3_auroc = plot(ax0, path, seed, legend=(True if i==0 else False))
        ds1_auroc_list.append(ds1_auroc)
        ds3_auroc_list.append(ds3_auroc)
    ds1_auroc_list = numpy.array(ds1_auroc_list)
    ds3_auroc_list = numpy.array(ds3_auroc_list)
    print("Dataset I AUROC +/- std: ", ds1_auroc_list.mean(), ds1_auroc_list.std(ddof=1))
    print("Dataset III AUROC +/- std: ", ds3_auroc_list.mean(), ds3_auroc_list.std(ddof=1))

    ds1_auroc_list = []
    ds3_auroc_list = []
    for i, (path, seed) in enumerate(zip(ds3_checkpoints, ds3_seeds)):
        ds1_auroc, ds3_auroc = plot(ax1, path, seed, legend=(True if i==0 else False))
        ds1_auroc_list.append(ds1_auroc)
        ds3_auroc_list.append(ds3_auroc)
    ds1_auroc_list = numpy.array(ds1_auroc_list)
    ds3_auroc_list = numpy.array(ds3_auroc_list)
    print("Dataset I AUROC +/- std: ", ds1_auroc_list.mean(), ds1_auroc_list.std(ddof=1))
    print("Dataset III AUROC +/- std: ", ds3_auroc_list.mean(), ds3_auroc_list.std(ddof=1))

    ax0.set_ylabel('True positive rate')
    dummy_ax = fig.add_subplot(111, frameon=False)
    dummy_ax.set_xlabel('False positive rate')
    dummy_ax.set_xticks([])
    dummy_ax.set_yticks([])
    for ax in [ax0, ax1]:
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect('equal')
        ax.set_xticks(numpy.linspace(0,1,6))
        ax.set_yticks(numpy.linspace(0,1,6))
        for kw in ['top', 'right']:
            ax.spines[kw].set_visible(False)
    ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.3, 0.4))
    ax1.set_yticklabels(['' for i in range(len(ax1.get_yticklabels()))])
    ax0.set_title('ChestX-ray14/\nGitHub-COVID', fontsize=6)
    ax1.set_title('BIMCV-COVID-19−/\nBIMCV-COVID-19+', fontsize=6)
    pyplot.savefig('roc_ds3.pdf')

if __name__ == "__main__":
    main()
