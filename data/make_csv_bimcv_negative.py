#!/usr/bin/env python
import json
import pandas
import os

from bimcv_negative_manualinfo import EXCLUDE, ENFORCE_LATERAL
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
        'TORAX  PA': 'PA',
        'TORAX  AP': 'AP',
        'TÓRAX AP PORTATIL': 'AP',
        'AP DIRECTO': 'AP',
        'PEDIÁTRICO AP': 'AP',
        'LATERAL DER.': 'LAT',
        'T035 TÓRAX AP': 'AP',
        'TORAX AP PARED': 'AP',
        'PARRILLA COSTAL AP': 'AP',
        'TORAX PEDIATRICO PA': 'PA',
        'ERECT AP': 'AP',
        'ABDOMEN': 'UNK',
        'TORAX FRENTE Y PERFIL': 'UNK',
        'TORAX LAT PARED': 'LAT',
        'RX TORAX  PA PED': 'PA',
        'SEDESTACION AP': 'AP',
        'PA HORIZONTAL': 'UNK',
        'AP VERTICAL': 'AP',
        'TÓRAX AP': 'AP',
        'LUMBAR-SPINE': 'UNK',
        'SIMPLE AP': 'AP',
        'TORAX PEDIATRICO AP': 'AP',
        'LAT': 'LAT',
        'TORAX NINOS DIRECTO AP': 'AP',
        'TORAX.PEDI PA': 'PA'}


def main():
    EXCLUDED_NO_PROJECTION = 0
    EXCLUDED_NO_WINDOW = 0
    datapath = 'bimcv-'
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
                            except (OSError, json.decoder.JSONDecodeError):
                                entry['projection'] = 'UNK'
                                data[idx] = entry
                                EXCLUDED_NO_PROJECTION += 1
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
                                EXCLUDED_NO_WINDOW += 1
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
                            imagename = path.strip().split('/')[-1]
                            if imagename in ENFORCE_LATERAL:
                                print("enforcing lateral projection for {:s}".format(imagename))
                                entry['projection'] = 'LAT'
                            if not imagename in EXCLUDE:
                                data[idx] = entry
                            else:
                                print('excluding ', imagename)

    df = pandas.DataFrame.from_dict(data, orient='index')
    df.to_csv(os.path.join(datapath, 'BIMCV-COVID-19-negative_NEW.csv'))

if __name__ == "__main__":
    main()
