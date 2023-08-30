import pandas as pd
import os
import sys
from tqdm import tqdm
import numpy as np
import datetime
import warnings
from omegaconf import DictConfig

sys.path.append(os.getcwd())
from src.pickle_manager import save_pickle


def num_to_date(n: int) -> str:
    """Converts 2-byte integer to a date string

    Args:
        d (int): 2-byte integer

    Returns:
        str: date string in the format 'YYYY-MM-DD'
    """
    year = ((n >> 9) & 0x7F) + 2000
    month = (n >> 5) & 0x0F
    day = n & 0x1F
    return f'{year:04}-{month:02}-{day:02}'


def load_egrul(cfg: DictConfig) -> pd.DataFrame:
    """Collects data from all .csv files with egrul data and creates one .pkl
    Remove OGRN duplicats
    """
    print("Loading EGRUL...")
    files = [cfg.paths.egrul + file for file in os.listdir(path=cfg.paths.egrul)]
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df.drop_duplicates(subset=['ogrn'], keep='first')
    
    # Drop unnecessary columns
    df = df.drop(
        columns=['crc32', 'kpp', 'short_name', 'email', 'pfr', 'fss', 'kladr','index', 'region', 'area', 'city',
        'settlement', 'street', 'house', 'corpus', 'apartment'] )

    # Convert 2bit date to date
    df['min_num'] = df['min_num'].map(num_to_date)
    df['max_num'] = df['max_num'].map(num_to_date)

    # Replace 0000-00-00 with NaN
    df.loc[df['end_date']=='0000-00-00', "end_date"] = np.nan
    
    # Replace all dates with datetime format
    df['reg_date'] = pd.to_datetime(df['reg_date'], dayfirst=True)
    df['end_date'] = pd.to_datetime(df['end_date'], dayfirst=True)
    df['min_num'] = pd.to_datetime(df['min_num'], dayfirst=True)
    df['max_num'] = pd.to_datetime(df['max_num'], dayfirst=True)

    # Drop very new companies (should we?)
    df = df.drop(df[df["reg_date"] > datetime.datetime(2023, 1, 1)].index)
    return df

def load_msp(cfg: DictConfig) -> pd.DataFrame:
    """Collects data from all .xlsx files with msp data and creates one .pkl
    """
    # Define the list of column names to select
    print("Loading MSP...")
    columns_to_select = ['Тип субъекта',
                        'Наименование / ФИО',
                        'Основной вид деятельности', 'Регион', 'Вновь созданный',
                        'Дата включения в реестр', 'Дата исключения из реестра',
                        'Наличие лицензий',
                        'ОГРН', 'ИНН'
                        ]

    # Initialize an empty DataFrame to hold the extracted data
    df_msp = pd.DataFrame()

    # Loop through all the files in the directory with the .xlsx extension
    for filename in tqdm(os.listdir(cfg.paths.msp)):
        if filename.endswith('.xlsx'):
            # Load the Excel file into a pandas DataFrame
            print(filename)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                df = pd.read_excel(os.path.join(cfg.paths.msp, filename), header=0, skiprows=2)
            # Select only the columns from the list
            df = df[columns_to_select]
            # Append the data to the combined DataFrame
            df_msp = pd.concat([df_msp, df], ignore_index=True)
    return df_msp


def load_data(cfg: DictConfig):
    df_egrul = load_egrul(cfg)
    save_pickle(cfg.paths.pkls, cfg.files.file_egrul, df_egrul)
    df_msp = load_msp(cfg)
    save_pickle(cfg.paths.pkls, cfg.files.file_msp, df_msp)


if __name__ == "__main__":
    load_data()