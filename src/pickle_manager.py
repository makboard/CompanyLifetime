import pickle
import os
import pandas as pd


def save_pickle(path: str, file_name: str, data: any) -> None:
    """Save data as pickle file"""
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), "wb") as fp:
        pickle.dump(data, fp)


def open_pickle(path: str, file_name: str) -> any:
    """Load data from pickle file"""
    with open(os.path.join(path, file_name), "rb") as fp:
        df = pickle.load(fp)
    return df


def save_parquet(path: str, file_name: str, data: pd.DataFrame) -> None:
    """Save data as pickle file"""
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_parquet(os.path.join(path, file_name))


def open_parquet(path: str, file_name: str) -> any:
    """Load data from pickle file"""
    df = pd.read_parquet(os.path.join(path, file_name))
    return df
