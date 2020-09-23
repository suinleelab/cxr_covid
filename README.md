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
While we provide code to load radiographs and associated metadata for training a deep-learning model, you will first need to download images from the above repositories. Organize the downloaded data as follows:

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
            derivatives/
                labels/
                    labels_covid19_posi.tsv
            sub-S0*/
                (subdirectories containing png images and json metadata)

Since the json files that contain metadata regarding the BIMCV-COVID-19+ radiographs can be unwieldy to work with, we next parse them to create a csv file that contains key metadata:

    cd ./data
    python make_csv.py 

For improved data loading performance, you may optionally create an HDF5 file for each image repository.


    cd ./data
    python make_h5.py 

## Training the models
