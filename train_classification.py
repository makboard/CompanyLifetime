from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os
import sys

import hydra
from omegaconf import DictConfig
from typing import Literal

import warnings

from src.pickle_manager import save_pickle, open_parquet
from src.classification_models import (
    logistic_regression_classification,
    random_forest_classification,
    xgb_classification,
)

warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())


def make_indices(X: pd.DataFrame) -> tuple[list, list, list]:
    """Split features indices into binary, numerical and categorical"""
    binary_columns = ["Тип субъекта", "Вновь созданный", "Наличие лицензий"]
    binary_indices = np.array(
        [(column in binary_columns) for column in X.columns], dtype=bool
    )

    categorical_columns = ["Основной вид деятельности", "Регион", "КатСубМСП"]
    categorical_indices = np.array(
        [(column in categorical_columns) for column in X.columns], dtype=bool
    )
    numeric_indices = ~(binary_indices + categorical_indices)
    numerical_columns = X.columns[numeric_indices]

    return binary_columns, categorical_columns, numerical_columns


def preprocess_X(X: pd.DataFrame, scaler = None, enc = None) -> pd.DataFrame:
    """Perform scaling and encoding for particular columns"""
    binary_columns, categorical_columns, numerical_columns = make_indices(X)
    X_cat = X[categorical_columns]
    X_num = X[numerical_columns]
    if scaler == None:
        # Scale numerical
        scaler = preprocessing.StandardScaler(with_mean=0)
        scaler.fit(X_num)
    X_num = pd.DataFrame(scaler.transform(X_num), columns=numerical_columns)

    if enc == None:
    # Encode categorical
        enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
        enc.fit(X_cat)
    X_cat = pd.DataFrame(
        enc.transform(X_cat).toarray().astype(np.int8), columns=enc.get_feature_names_out()
    )

    # Concat all columns
    X = pd.concat(
        [
            X[binary_columns].reset_index(drop=True),
            X_num.reset_index(drop=True),
            X_cat.reset_index(drop=True),
        ],
        axis=1,
    )

    return X, scaler, enc

def classify_lifetime(lifetime) -> Literal[0, 1, 2, 3, 4] | None:
    if lifetime <= 12:
        return 0
    elif 12 < lifetime <= 24:
        return 1
    elif 24 < lifetime <= 48:
        return 2
    elif 48 < lifetime <= 120:
        return 3
    elif lifetime > 120:
        return 4

def train_test(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform train/test split"""
    print("Collecting data...")
    data = open_parquet(cfg.paths.parquets, cfg.files.companies_feat)
    print("Initial data shape:", data.shape)

    data = data.dropna()
    print("Final data shape:", data.shape)

    data.drop(data.loc[data["Основной вид деятельности"] == "na"].index, inplace=True)
    data.drop(data.loc[data["Основной вид деятельности"] == "No"].index, inplace=True)
    data['lifetime'] = data['lifetime'].apply(classify_lifetime)
    y = data["lifetime"].values
    X = data.drop(["Наименование / ФИО", "ОГРН", "ИНН", "reg_date", "lifetime"], axis=1)

    # Cut values in some columns
    X["Основной вид деятельности"] = X.apply(
        lambda row: row["Основной вид деятельности"][1:2]
        if row["Основной вид деятельности"][0] == "0"
        else row["Основной вид деятельности"][:2],
        axis=1,
    )
    X["Регион"] = X.apply(
        lambda row: row["Регион"][1:2]
        if row["Регион"][0] == "0"
        else row["Регион"][:2],
        axis=1,
    )

    # Convert columns data types
    X["Тип субъекта"] = (X["Тип субъекта"] == "Индивидуальный предприниматель").astype(
        int
    )
    X[["Основной вид деятельности", "Регион"]] = X[
        ["Основной вид деятельности", "Регион"]
    ].astype(int)
    X["Вновь созданный"] = (X["Вновь созданный"] == "Да").astype(int)
    X["Наличие лицензий"] = (X["Наличие лицензий"] == "Да").astype(int)
    X["КатСубМСП"] = X["КатСубМСП"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    X_train, scaler, enc = preprocess_X(X_train)
    X_test, _, _ = preprocess_X(X_test, scaler, enc)
    save_pickle(cfg.paths.pkls, cfg.files.num_scaler, scaler)
    save_pickle(cfg.paths.pkls, cfg.files.cat_enc, enc)
    return X_train, X_test, y_train, y_test


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="classification_config.yaml")
def run_train(cfg: DictConfig):
    metrics_dict = {}
    X_train, X_test, y_train, y_test = train_test(cfg)
    save_pickle(cfg.paths.pkls, cfg.files.processed_dataset, data=(X_train, X_test, y_train, y_test))
    metrics_dict = xgb_classification(
        cfg, X_train, y_train, X_test, y_test, metrics_dict
    )
    metrics_dict = random_forest_classification(
        cfg, X_train, y_train, X_test, y_test, metrics_dict
    )
    metrics_dict = logistic_regression_classification(
        cfg, X_train, y_train, X_test, y_test, metrics_dict
    )
    
    models = []
    frames = []

    for model_id, d in metrics_dict.items():
        models.append(model_id)
        frames.append(pd.DataFrame.from_dict(d, orient="index"))
    metrics_table = pd.concat(frames, keys=models)
    metrics_table = metrics_table.reset_index()
    print(metrics_table)
    save_pickle(cfg.paths.models, cfg.files.metrics, metrics_dict)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_train()