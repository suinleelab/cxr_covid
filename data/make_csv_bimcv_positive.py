#!/usr/bin/env python
import json
import pandas
import os
series_description_map = {
        'TORAX AP': 'AP',
        'PORTATIL': 'AP',
        'CHEST': 'UNK',
        'W034 TÓRAX LAT.': 'LAT',
        'AP HORIZONTAL': 'AP SUPINE',
        'TÓRAX PA H': 'PA',
        'BUCKY PA': 'PA',
        'ESCAPULA Y': 'UNK',
        'LATERAL IZQ.': 'LAT',
        'TORAX SUPINE AP': 'AP SUPINE',
        'DIRECTO AP': 'AP',
        'T034 TÓRAX LAT': 'LAT',
        'PECHO AP': 'AP',
        'TORAX AP DIRECTO': 'AP',
        'W034 TÓRAX LAT. *': 'LAT',
        'TÓRAX LAT': 'LAT',
        'ERECT LAT': 'LAT',
        'TORAX LAT': 'LAT',
        'TÓRAX AP H': 'AP SUPINE',
        'TÒRAX AP': 'AP',
        'TORAX PORTATIL': 'AP',
        'DEC. SUPINO AP': 'AP SUPINE',
        'SUPINE AP': 'AP SUPINE',
        'TÓRAX': 'UNK',
        'RX TORAX CON PORTATIL': 'AP',
        'TORAX PA': 'PA',
        'TORAX ERECT PA': 'PA',
        'DIRECTO PA': 'PA',
        'RX TORAX CON PORTATIL PED': 'AP',
        'LATERAL': 'LAT',
        'TORAX BIPE PA': 'PA',
        'SUP.AP PORTABLE': 'AP SUPINE', 
        'TORAX CAMILLA': 'AP',
        'TORAX-ABD PA': 'PA',
        'TORAX SEDE AP': 'AP',
        'BUCKY LAT': 'LAT',
        'ERECT PA': 'PA',
        'TORAX SUPINO AP': 'AP SUPINE',
        'W033 TÓRAX AP': 'AP',
        'PORTÁTIL AP': 'AP',
        'TORAX ERECT LAT': 'LAT',
        'PA': 'PA',
        'W033 TÓRAX PA *': 'PA',
        'TÓRAX PA': 'PA',
        'TÃ²RAX AP': 'PA',
        'RX TORAX  PA Y LAT': 'UNK',
        'AP': 'AP', 
        'T035 TÓRAX PA': 'PA', 
        'RX TORAX, PA O AP': 'UNK', 
        'W033 TÓRAX PA': 'PA', 
        'TORAX  PA': 'PA'}

ENFORCE_LATERAL = [
        "bimcv+/sub-S04079/ses-E08254/mod-rx/sub-S04079_ses-E08254_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03582/ses-E07281/mod-rx/sub-S03582_ses-E07281_run-1_bp-chest_vp-ap_cr.png",
        "bimcv+/sub-S03585/ses-E07287/mod-rx/sub-S03585_ses-E07287_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03585/ses-E07911/mod-rx/sub-S03585_ses-E07911_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03996/ses-E08157/mod-rx/sub-S03996_ses-E08157_run-1_bp-chest_vp-ap_cr.png",
        "bimcv+/sub-S04334/ses-E08628/mod-rx/sub-S04334_ses-E08628_acq-2_run-1_bp-chest_vp-pa_cr.png",
        "bimcv+/sub-S04489/ses-E08918/mod-rx/sub-S04489_ses-E08918_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04224/ses-E08468/mod-rx/sub-S04224_ses-E08468_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04084/ses-E08261/mod-rx/sub-S04084_ses-E08261_acq-2_run-1_bp-chest_vp-pa_cr.png",
        "bimcv+/sub-S03898/ses-E07949/mod-rx/sub-S03898_ses-E07949_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04190/ses-E08422/mod-rx/sub-S04190_ses-E08422_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04195/ses-E08429/mod-rx/sub-S04195_ses-E08429_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04075/ses-E08435/mod-rx/sub-S04075_ses-E08435_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04075/ses-E08250/mod-rx/sub-S04075_ses-E08250_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04484/ses-E08905/mod-rx/sub-S04484_ses-E08905_run-1_bp-chest_vp-ap_cr.png",
        "bimcv+/sub-S03736/ses-E07772/mod-rx/sub-S03736_ses-E07772_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04021/ses-E08189/mod-rx/sub-S04021_ses-E08189_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04275/ses-E08532/mod-rx/sub-S04275_ses-E08532_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03939/ses-E08092/mod-rx/sub-S03939_ses-E08092_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04298/ses-E08566/mod-rx/sub-S04298_ses-E08566_acq-2_run-1_bp-chest_vp-pa_cr.png",
        "bimcv+/sub-S04101/ses-E08526/mod-rx/sub-S04101_ses-E08526_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04101/ses-E08453/mod-rx/sub-S04101_ses-E08453_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03699/ses-E07505/mod-rx/sub-S03699_ses-E07505_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04203/ses-E08442/mod-rx/sub-S04203_ses-E08442_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04316/ses-E08597/mod-rx/sub-S04316_ses-E08597_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03610/ses-E07319/mod-rx/sub-S03610_ses-E07319_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S04402/ses-E08747/mod-rx/sub-S04402_ses-E08747_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03563/ses-E07251/mod-rx/sub-S03563_ses-E07251_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03936/ses-E08089/mod-rx/sub-S03936_ses-E08089_acq-2_run-1_bp-chest_vp-pa_dx.png",
        "bimcv+/sub-S03931/ses-E08648/mod-rx/sub-S03931_ses-E08648_acq-2_run-1_bp-chest_vp-pa_dx.png"
        ]

def main():
    datapath = 'bimcv+'
    patientdf = pandas.read_csv(os.path.join(datapath, 'participants.tsv'),
                              sep='\t')

    data = {}
    series_descriptions = set()
    idx = -1 
    for _, row in patientdf.iterrows():
        subject = row.participant 
        modalities = row.modality_dicom
        modalities = eval(modalities)
        if 'CR' in modalities or 'DX' in modalities:
            contents = os.listdir(os.path.join(datapath, subject))
            for sessionfile in contents:
                if os.path.isdir(os.path.join(datapath, subject, sessionfile)):
                    image_candidates = os.listdir(os.path.join(datapath, subject, sessionfile, 'mod-rx'))
                    for i in image_candidates:
                        if i.lower().endswith('.png'):
                            idx += 1
                            entry = {}
                            path = os.path.join(datapath, subject, sessionfile, 'mod-rx', i)
                            entry['path'] = path
                            entry['participant'] = subject
                            jsonpath = path[:-4] + '.json'
                            try:
                                with open(jsonpath, 'r') as handle:
                                    metadata = json.load(handle)
                            except OSError:
                                entry['projection'] = 'UNK'
                                data[idx] = entry
                                break
                            entry['modality'] = metadata['00080060']['Value'][0]
                            entry['manufacturer'] = metadata['00080070']['Value'][0]
                            entry['sex'] = metadata['00100040']['Value'][0]
                            try:
                                photometric_interpretation = metadata['00280004']['Value'][0]
                                entry['photometric_interpretation'] = photometric_interpretation
                            except KeyError:
                                print('no photometric_interpretation for: ', path)
                            try:
                                entry['rotation'] = metadata['00181140']['Value'][0]
                                print(entry['rotation'])
                            except KeyError:
                                pass
                            try:
                                entry['lut'] = metadata['00283010']['Value'][0]['00283006']['Value']
                                entry['lut_min'] = metadata['00283010']['Value'][0]['00283002']['Value'][1]
                                try:
                                    entry['rescale_slope'] = metadata['00281053']['Value'][0]
                                    entry['rescale_intercept'] = metadata['00281052']['Value'][0]
                                except KeyError:
                                    pass
                                try:
                                    entry['bits_stored'] = metadata['00280101']['Value'][0]
                                except KeyError:
                                    try: 
                                        entry['bits_stored'] = metadata['00283010']['Value'][0]['00283002']['Value'][2]
                                    except KeyError:
                                        pass

                            except KeyError:
                                try:
                                    entry['window_center'] = metadata['00281050']['Value'][0]
                                    entry['window_width'] = metadata['00281051']['Value'][0]
                                except KeyError:
                                    print("No window information for : ", path)
                            try: 
                                entry['study_date'] = int(metadata['00080020']['Value'][0])
                            except KeyError:
                                pass
                            try:
                                entry['study_time'] = float(metadata['00080030']['Value'][0])
                            except KeyError:
                                pass
                            try:
                                entry['age'] = int(metadata['00101010']['Value'][0][:-1])
                            except KeyError:
                                pass
                            try:
                                series_description = metadata['0008103E']['Value'][0]
                            except Exception as e:
                                try:
                                    series_description = metadata['00081032']['Value'][0]['00080104']['Value'][0]
                                except Exception as e:
                                    raise e
                            series_description = series_description.upper()
                            series_descriptions.add(series_description)
                            projection = series_description_map[series_description]
                            entry['projection'] = projection

                            # these images are manually set to lateral
                            if path.strip() in ENFORCE_LATERAL:
                                print("enforcing lateral projection for {:s}".format(path))
                                entry['projection'] = 'LAT'
                            data[idx] = entry

    df = pandas.DataFrame.from_dict(data, orient='index')
    df.to_csv(os.path.join(datapath, 'bimcv+.csv'))

if __name__ == "__main__":
    main()
