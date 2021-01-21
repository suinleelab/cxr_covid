#!/usr/bin/env python
from datasets.bimcvcovid import *

class BIMCVNegativeDataset(BIMCVCOVIDDataset):
    def __init__(self, fold, random_state=30493, include_lateral=False, 
                 include_unknown_projections=False, include_ap_supine=False,
                 include_unknown_labels=False, initialize_h5=False, covid_labels='molecular',
                 labels='chexpert'):
        if include_unknown_projections:
            raise NotImplementedError('not implemented for bimcv negative dataset')
        super().__init__(fold, random_state=random_state, include_lateral=include_lateral,
                include_unknown_projections=include_unknown_projections, include_ap_supine=include_ap_supine,
                include_unknown_labels=include_unknown_labels, initialize_h5=initialize_h5, covid_labels=covid_labels,
                labels=labels)

    def _set_datapaths(self):
        self.datapath = 'data/bimcv-'
        self.labelpath = 'derivatives/labels/labels_SARS-cov-2_nega.tsv'
        # leave the same as bimcv-covid-19+ for now
        self.unknown_label_path = 'datasets/bimcv_covid_unknown_labels.txt'
        self.labeldf = pandas.read_csv(os.path.join(self.datapath, self.labelpath), delimiter='\t')
        self.h5path = 'data/bimcv-/bimcv-.h5'
        self.df = pandas.read_csv(os.path.join(self.datapath, 'BIMCV-COVID-19-negative.csv'))
        # Not available for now
        self.manual_projection_label_path = "datasets/bimcv_covid_manual_projection_labels.yml"

    def add_covid_label(self, findings):
        # NO samples are covid-19+
        return findings
