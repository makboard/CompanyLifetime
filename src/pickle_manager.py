import pickle
import os
import pandas as pd
from typing import Any


def save_pickle(path: str, file_name: str, data: Any) -> None:
    """
    Saves the given data to a file in pickle format.

    Parameters:
    path (str): Directory path to save the file.
    file_name (str): Name of the file to be saved.
    data (Any): The data to be saved.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, file_name), "wb") as file:
        pickle.dump(data, file)


def open_pickle(path: str, file_name: str) -> Any:
    """
    Loads data from a pickle file.

    Parameters:
    path (str): Directory path where the file is located.
    file_name (str): Name of the file to be loaded.

    Returns:
    Any: The data loaded from the pickle file.
    """
    with open(os.path.join(path, file_name), "rb") as file:
        return pickle.load(file)


def save_parquet(path: str, file_name: str, data: pd.DataFrame) -> None:
    """
    Saves a DataFrame to a file in Parquet format.

    Parameters:
    path (str): Directory path to save the file.
    file_name (str): Name of the file to be saved.
    data (pd.DataFrame): DataFrame to be saved.
    """
    os.makedirs(path, exist_ok=True)
    data.to_parquet(os.path.join(path, file_name))


def open_parquet(path: str, file_name: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a Parquet file.

    Parameters:
    path (str): Directory path where the file is located.
    file_name (str): Name of the file to be loaded.

    Returns:
    pd.DataFrame: The DataFrame loaded from the Parquet file.
    """
    return pd.read_parquet(os.path.join(path, file_name))
