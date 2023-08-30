## Data sources

Data sources include:
* egrul folder with `.csv` files listing all russian companies. Loaded for each region from [Nalog.ru](https://egrul.nalog.ru/index.html);
* rosstat folder with `.xlsx` files aggregating regional statistics from [Rosstat.gov.ru](https://rosstat.gov.ru/folder/210/document/13204). Version 2022;
* msp folder with `.xlsx` files from [Nalog.ru](https://rmsp.nalog.ru/search.html?mode=extended). Includes small and middle companies only. HERE edit the version date used (see this parameter at mentioned page);
* msp_xml folder with `.xml` files from [Nalog.ru](https://www.nalog.gov.ru/opendata/7707329152-rsmp/). Data very similar to previous one, but with different representations and few extra features included.
* features.xlsx file created by hand. It lists all regional features used in analysis.

Further processing saves intermidiate files to `data` folder.

`Config` folder stores configurations for data and models' parameters. These parameters were obtained with Optuna optimization (not included).

## Docker

From repo folder run:

* `docker build -t stat .`
* `docker run -it  -v  <CODE FOLDER>:/workdir -v <DATA FOLDER>:/workdir/ -m 16000m  --cpus=4  -w="/workdir" stat`

## Loading data

There are 2 options:

* Inside the container run `.sh` (*not implemented yet*) with raw company data (`.xls`, `.xlsx`, `.csv` file formats) from google drive, unpack it, delete the archived data.
* Inside the container run `sh download.sh` -- to download preprocessed company data (`.pkl` file format) from google drive, unpack it, delete the archived data.

## Executing program

* preprocess.py handles raw data. Thus, it works after loading data with option 1. Output file `data/pkls/companies_feat.pkl` contains all companies mentioned in MSP registry and closed up to date (i.e. companies with finite 'lifetime' feature serving as a target variable).
This step requires `data_raw` folder. However you may skip it and use `.pkl` files loaded to folder `data` with option 2.
* train.py performs the regression analysis with several algorythms and writes pretrained models and their metrics in `data/models/metrics.pkl` file.
* run.py (*not implemented yet*) predicts lifetime for a company with parameters listed in `config\predict.yaml`