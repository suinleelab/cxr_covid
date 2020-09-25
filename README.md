# cxr_covid
Code for paper "AI for radiographic COVID-19 detection selects shortcuts over signal"<br/>
<br/>
Datasets can be downloaded at the following links:<br/>
**Dataset I**<br/>
[Cohen et al. Covid-Chestxray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset)<br/>
[ChestXray-14](https://nihcc.app.box.com/v/ChestXray-NIHCC)<br/>
<br/>
**Dataset II**<br/>
[BIMCV-COVID-19 +](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/)<br/>
[PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/)<br/>

## Setting up the datasets
While we provide code to load radiographs and associated metadata for training a deep-learning model, you will first need to download images from the above repositories. Be aware that these repositories combine to multiple terabytes of data. 

Organize the downloaded data as follows:

    ./data/
        ChestX-ray14/
            labels/
                Data_Entry_2017.csv
                test_list.txt
                train_val_list.txt
            images/
                (many image files)
        GitHub-COVID/
            metadata.csv
            images/
                (many image files)
        PadChest/
            PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
            images/
                (many image files)
        BIMCV-COVID-19
            participants.tsv
            derivatives/
                labels/
                    labels_covid19_posi.tsv
            sub-S0*/
                (subdirectories containing png images and json metadata)

### ChestX-ray14
Download the files listed above under `./data/ChestX-ray14/.` You will need to download and extract all of the zip files from the images directory and organize all of the images into a single directory (`./data/Chestx-ray14/images`). Note that some file names may change (e.g., `Data_Entry_2017.csv` may have been renamed to `Data_Entry_2017_v2020.csv` depending on your download date). It is important that you rename files to match the above scheme.

### Cohen et al. Covid-Chestxray-Dataset (a.k.a. "GitHub-COVID" in our manuscript)
Simply clone the repository and rename it as `./data/GitHub-COVID`.

### PadChest
Download each of the image zip files as well as the csv file containing metadata. Extract all of the images and organize them into a single directory at `./data/PadChest/images`.

### BIMCV-COVID19+
Download all of the zip files, which contain both the images and metadata. Place all of the zip files in `./data/BIMCV-COVID-19` and extract them. You should end up with a subdirectory named "derivatives" which includes some of the metadata, as well as many folders named "sub-SXXXXX" (where XXXXX is a number) which contain the images and more metadata.

Since the json files that contain metadata regarding the BIMCV-COVID-19+ radiographs can be unwieldy to work with, parse them to create a csv file that contains key metadata:

    cd ./data
    python make_csv.py 

### HDF5 Files

For improved data loading performance, create an HDF5 files for the image repositories. Note that due to its small size, we do not provide scripts for loading the GitHub-COVID from HDF5 files.

To generate the files, run the following commands:

    cd ./data
    python make_h5.py -i ChestX-ray14 -o ChestX-ray14/chestxray14.h5
    python make_h5.py -i PadChest -o PadChest/padchest.h5
    python make_h5.py -i BIMCV-COVID-19 -o BIMCV-COVID-19/BIMCV-COVID-19.h5 

Check to make sure the output files are organized as follows:

    data/
        ChestX-ray14/
            chestxray14.h5
        PadChest/
            padchest.h5
        BIMCV-COVID-19
            BIMCV-COVID-19.h5

## Training the models
After setting up the datasets, train models using the `train_covid.py` script. This script works via the command line; for more information on using the script, run `python train_covid.py --help`. 

## Evaluating the models
Once you have trained models on both datasets, evaluate the models using the script `roc.py`. This will calculate receiver operating characteristic curves for both internal and external test data. First, edit the "options" section of `roc.py` to match the output paths from model training; the checkpoint files may be found in `./checkpoints`. Then, call `python roc.py` to generate the ROC curves. 
