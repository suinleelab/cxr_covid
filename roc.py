#!/usr/bin/env python
#
# roc.py
#
# Script for calculating ROC curves for internal and external test data. 
#
# To run, modify the paths below in the "Options" section. Then call
# ``python roc.py''.  Output will be saved to roc.pdf.
#

import numpy
import sklearn.metrics

from models import CXRClassifier
from datasets import ChestXray14H5Dataset, PadChestH5Dataset
from datasets import GitHubCOVIDDataset, BIMCVCOVIDDataset
from datasets import DomainConfoundedDataset
from train_covid import _find_index
import matplotlib
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
import matplotlib.pyplot as pyplot

import os
import sys

#################################### Options ##################################
# Modify to match the output files from your training procedure. The random
# seeds are automatically parsed from the file names.

# Models trained on dataset I
ds1_checkpoints = \
        ['checkpoints/githubcxr14.densenet121.30493.pkl.best_auroc',
         'checkpoints/githubcxr14.densenet121.30494.pkl.best_auroc',
         'checkpoints/githubcxr14.densenet121.30495.pkl.best_auroc',
         'checkpoints/githubcxr14.densenet121.30496.pkl.best_auroc',
         'checkpoints/githubcxr14.densenet121.30497.pkl.best_auroc']
# Models trained on dataset II
ds2_checkpoints = \
        ['checkpoints/bimcvpadchest.densenet121.30493.pkl.best_auroc',
         'checkpoints/bimcvpadchest.densenet121.30494.pkl.best_auroc',
         'checkpoints/bimcvpadchest.densenet121.30495.pkl.best_auroc',
         'checkpoints/bimcvpadchest.densenet121.30496.pkl.best_auroc',
         'checkpoints/bimcvpadchest.densenet121.30497.pkl.best_auroc']
###############################################################################


def plot(ax, checkpointpath, seed, legend=False):
    githubcxr14_testds = DomainConfoundedDataset(
            ChestXray14H5Dataset(fold='test', labels='chestx-ray14', random_state=seed),
            GitHubCOVIDDataset(fold='test', labels='chestx-ray14', random_state=seed)
            )

    bimcvpadchest_testds = DomainConfoundedDataset(
            PadChestH5Dataset(fold='test', labels='chestx-ray14', random_state=seed),
            BIMCVCOVIDDataset(fold='test', labels='chestx-ray14', random_state=seed)
            )

    classifier = CXRClassifier()
    classifier.load_checkpoint(checkpointpath)
    
    githubcxr14_probs = classifier.predict(githubcxr14_testds)
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

    bimcvpadchest_probs = classifier.predict(bimcvpadchest_testds)
    bimcvpadchest_true = bimcvpadchest_testds.get_all_labels()
    bimcvpadchest_idx = _find_index(bimcvpadchest_testds, 'COVID')
    bimcvpadchest_auroc = sklearn.metrics.roc_auc_score(
            bimcvpadchest_true[:, bimcvpadchest_idx],
            bimcvpadchest_probs[:, githubcxr14_idx] 
            )
    print("bimcvpadchest auroc: ", bimcvpadchest_auroc)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
            bimcvpadchest_true[:, bimcvpadchest_idx],
            bimcvpadchest_probs[:, bimcvpadchest_idx] 
            )
    kwargs = {'color': '#107f80', 'linewidth': 1}
    if legend: ax.plot(fpr, tpr, label='PadChest/\nBIMCV-COVID-19+', **kwargs)
    else: ax.plot(fpr, tpr, **kwargs)
    return githubcxr14_auroc, bimcvpadchest_auroc

def main():
    ds1_seeds = [int(os.path.basename(filepath).split('.')[2]) \
                 for filepath in ds1_checkpoints]

    ds2_seeds = [int(os.path.basename(filepath).split('.')[2]) \
                 for filepath in ds2_checkpoints]

    fig = pyplot.gcf()
    fig.set_size_inches(8.8/2.54, 2)
    ax0 = pyplot.subplot(1, 2, 1)
    ax1 = pyplot.subplot(1, 2, 2)
    for ax in [ax0, ax1]:
        ax.plot((0,1), (0,1), color='#a0a0a0', linewidth=1, ls='--')

    # Calculations for models trained on dataset 1
    ds1_auroc_list = []
    ds2_auroc_list = []
    for i, (path, seed) in enumerate(zip(ds1_checkpoints, ds1_seeds)):
        ds1_auroc, ds2_auroc = plot(ax0, path, seed, legend=(True if i==0 else False))
        ds1_auroc_list.append(ds1_auroc)
        ds2_auroc_list.append(ds2_auroc)
    ds1_auroc_list = numpy.array(ds1_auroc_list)
    ds2_auroc_list = numpy.array(ds2_auroc_list)
    print("Statistics for models trained on dataset I:")
    print("Dataset I ROC-AUC +/- std: ", ds1_auroc_list.mean(), " +/- ", ds1_auroc_list.std(ddof=1))
    print("Dataset II ROC-AUC +/- std: ", ds2_auroc_list.mean(), " +/- ", ds2_auroc_list.std(ddof=1))

    # Calculations for models trained on dataset 2
    ds1_auroc_list = []
    ds2_auroc_list = []
    for i, (path, seed) in enumerate(zip(ds2_checkpoints, ds2_seeds)):
        ds1_auroc, ds2_auroc = plot(ax1, path, seed, legend=(True if i==0 else False))
        ds1_auroc_list.append(ds1_auroc)
        ds2_auroc_list.append(ds2_auroc)
    ds1_auroc_list = numpy.array(ds1_auroc_list)
    ds2_auroc_list = numpy.array(ds2_auroc_list)
    print("Statistics for models trained on dataset II:")
    print("Dataset I ROC-AUC +/- std: ", ds1_auroc_list.mean(), " +/- ", ds1_auroc_list.std(ddof=1))
    print("Dataset II ROC-AUC +/- std: ", ds2_auroc_list.mean(), " +/- ", ds2_auroc_list.std(ddof=1))

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
    ax0.set_title('Dataset I', fontsize=6)
    ax1.set_title('Dataset II', fontsize=6)
    pyplot.savefig('roc.pdf')

if __name__ == "__main__":
    main()
