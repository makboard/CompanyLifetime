import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn import preprocessing

from .classification_models import (logistic_regression,
                                    random_forest_classification,
                                    xgb_classification)
from .dataset_manager import DatasetManager
from .pickle_manager import open_parquet, save_parquet, save_pickle
from .regression_models import (linear_regression, ridge_regression,
                                xgb_regression)

logging.basicConfig(level=logging.INFO)


def make_indices(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits feature indices into binary, numerical, and categorical.

    Parameters:
    X (pd.DataFrame): DataFrame with features.

    Returns:
    Tuple[List[str], List[str], List[str]]:Lists of binary, categorical, and numerical column names.
    """
    binary_columns = ["Тип субъекта", "Вновь созданный", "Наличие лицензий"]
    categorical_columns = ["Основной вид деятельности", "Регион", "КатСубМСП"]
    binary_indices = df.columns.isin(binary_columns)
    categorical_indices = df.columns.isin(categorical_columns)
    numerical_columns = df.columns[~(binary_indices | categorical_indices)]

    return binary_columns, categorical_columns, numerical_columns.tolist()


def preprocess_features(
    df: pd.DataFrame,
    scaler: Optional[preprocessing.StandardScaler] = None,
    encoder: Optional[preprocessing.OneHotEncoder] = None,
) -> Tuple[pd.DataFrame, preprocessing.StandardScaler, preprocessing.OneHotEncoder]:
    """
    Performs scaling and encoding on specific columns.

    Parameters:
    df (pd.DataFrame): DataFrame with features to preprocess.
    scaler (preprocessing.StandardScaler, optional): Scaler for numerical features.
        Defaults to None.
    encoder (preprocessing.OneHotEncoder, optional): Encoder for categorical features.
        Defaults to None.

    Returns:
    Tuple[pd.DataFrame, preprocessing.StandardScaler, preprocessing.OneHotEncoder]:
        Preprocessed DataFrame, scaler, and encoder.
    """
    binary_columns, categorical_columns, numerical_columns = make_indices(df)
    df_cat = df[categorical_columns]
    df_num = df[numerical_columns]

    # Scale numerical columns
    if scaler is None:
        scaler = preprocessing.StandardScaler()
        scaler.fit(df_num)
    df_num = pd.DataFrame(scaler.transform(df_num), columns=numerical_columns)

    # Encode categorical columns
    if encoder is None:
        encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")
        encoder.fit(df_cat)
    df_cat = pd.DataFrame(
        encoder.transform(df_cat).toarray().astype(np.int8),
        columns=encoder.get_feature_names_out(),
    )

    # Concatenate all columns
    df_preprocessed = pd.concat(
        [
            df[binary_columns].reset_index(drop=True),
            df_num.reset_index(drop=True),
            df_cat.reset_index(drop=True),
        ],
        axis=1,
    )

    return df_preprocessed, scaler, encoder


def train_test(
    cfg: DictConfig,
    classification: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs train/test split on the dataset.

    Parameters:
    cfg (DictConfig): Configuration object.
    classification (bool): Specifies the model type. Defaults to False.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train and test splits for
        features and target.
    """
    logging.info("Collecting data...")
    data = open_parquet(cfg.paths.parquets, cfg.files.companies_feat)
    logging.info(f"Initial data shape: {data.shape}")

    data = data.dropna()
    logging.info(f"Final data shape: {data.shape}")

    y = data["lifetime"]
    X = data.drop(
        ["Наименование / ФИО", "ОГРН", "ИНН", "reg_date", "lifetime"],
        axis=1,
        errors="ignore",
    )

    # Modify specific columns
    for col in ["Основной вид деятельности", "Регион"]:
        X[col] = X[col].apply(lambda x: x[1:2] if x[0] == "0" else x[:2])

    # Convert categories to numerical representations
    category_mappings = {
        "Тип субъекта": {"Индивидуальный предприниматель": 1},
        "Вновь созданный": {"Да": 1},
        "Наличие лицензий": {"Да": 1},
    }
    for col, mapping in category_mappings.items():
        X[col] = X[col].map(mapping).fillna(0).astype(int)

    X["КатСубМСП"] = X["КатСубМСП"].astype(int)

    # Instantiate the DatasetManager class
    dataset_manager = DatasetManager()

    X_train, X_test, y_train, y_test = dataset_manager.fit_transform(
        df=X, y=y, classification=classification
    )

    dataset_manager.save_instance(os.path.join(cfg.paths.pkls, cfg.files.data_manager))
    return X_train, X_test, y_train, y_test


def save_metrics_to_parquet(cfg: DictConfig, metrics_dict: dict) -> None:
    """
    Saves the metrics dictionary to a Parquet file.

    Parameters:
    cfg (DictConfig): Configuration object containing paths and file names.
    metrics_dict (dict): Dictionary containing the metrics of different models.

    Description:
    This function takes a dictionary of metrics, where each key represents a model
    and its value is another dictionary of various performance metrics. It converts
    this nested dictionary into a Pandas DataFrame and then saves it as a Parquet file.
    """
    model_ids, frames = zip(
        *(
            (model_id, pd.DataFrame.from_dict(metrics, orient="index"))
            for model_id, metrics in metrics_dict.items()
        )
    )
    metrics_table = pd.concat(frames, keys=model_ids)
    metrics_table = metrics_table.reset_index(level=0).rename(
        columns={"level_0": "Model"}
    )

    # Ensure the output directory exists
    os.makedirs(cfg.paths.models, exist_ok=True)
    output_path = os.path.join(cfg.paths.models, cfg.files.metrics)
    save_parquet(cfg.paths.models, cfg.files.metrics, metrics_table)

    logging.info(f"Metrics saved to {output_path}")
    logging.info(metrics_table)


def update_config_filenames(
    cfg: DictConfig, model_type: str, suffix: str
) -> DictConfig:
    """
    Updates the file names in the configuration with a specific suffix.

    Parameters:
    cfg (DictConfig): Configuration object.
    model_type (str): Type of the model, e.g., 'classification' or 'regression'.
    suffix (str): Suffix to append to the file names.

    Returns:
    DictConfig: Updated configuration object.
    """
    for key in [
        "processed_dataset",
        "data_manager",
        "metrics",
    ]:
        cfg.files[key] = f"{key}_{model_type}_{suffix}.pkl"
    return cfg


def run_classification(cfg: DictConfig, suffix: str) -> None:
    """
    Runs the classification models and saves the metrics.

    Parameters:
    cfg (DictConfig): Configuration object.
    suffix (str): Suffix for file naming to differentiate the output files.
    """
    cfg = update_config_filenames(cfg, "classification", suffix)

    logging.info("Running classification models...")
    X_train, X_test, y_train, y_test = train_test(cfg, True)
    save_pickle(
        cfg.paths.pkls,
        cfg.files.processed_dataset,
        data=(X_train, X_test, y_train, y_test),
    )

    metrics_dict = {}
    for model_func in [
        logistic_regression,
        random_forest_classification,
        xgb_classification,
    ]:
        metrics_dict = model_func(cfg, X_train, y_train, X_test, y_test, metrics_dict)

    save_metrics_to_parquet(cfg, metrics_dict)


def run_regression(cfg: DictConfig, suffix: str) -> None:
    """
    Runs the regression models and saves the metrics.

    Parameters:
    cfg (DictConfig): Configuration object.
    suffix (str): Suffix for file naming to differentiate the output files.
    """
    cfg = update_config_filenames(cfg, "regression", suffix)

    print("Running regression models...")
    X_train, X_test, y_train, y_test = train_test(cfg)
    save_pickle(
        cfg.paths.pkls, cfg.files.processed_dataset, (X_train, X_test, y_train, y_test)
    )

    metrics_dict = {}
    for model_func in [linear_regression, ridge_regression, xgb_regression]:
        metrics_dict = model_func(cfg, X_train, y_train, X_test, y_test, metrics_dict)

    save_metrics_to_parquet(cfg, metrics_dict)


def update_filename(file_name: str, suffix: str) -> str:
    """
    Appends a suffix to the base name of a file while keeping its extension.

    Parameters:
    file_name (str): The original file name.
    suffix (str): The suffix to append to the file name.

    Returns:
    str: The updated file name with the suffix appended before the file extension.
    """
    base, ext = os.path.splitext(file_name)
    return f"{base}_{suffix}{ext}"
