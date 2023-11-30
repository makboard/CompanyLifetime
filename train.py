import os
import sys
import warnings
from typing import Literal, Tuple, List

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import hydra
from omegaconf import DictConfig

from src.pickle_manager import save_pickle, open_parquet, save_parquet
from src.classification_models import (
    logistic_regression_classification,
    random_forest_classification,
    xgb_classification,
    catboost_classification,
)
from src.regression_models import linear_regression, ridge_regression, xgb_regression, catboost_regression

# Configure warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())


def make_indices(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
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


def preprocess_features(X: pd.DataFrame, scaler=None, encoder=None) -> pd.DataFrame:
    """Perform scaling and encoding for particular columns"""
    binary_columns, categorical_columns, numerical_columns = make_indices(X)
    X_cat = X[categorical_columns]
    X_num = X[numerical_columns]
    # Scale numerical
    if scaler == None:
        scaler = preprocessing.StandardScaler(with_mean=0)
        scaler.fit(X_num)
    X_num = pd.DataFrame(scaler.transform(X_num), columns=numerical_columns)
    # Encode categorical
    if encoder == None:
        encoder = preprocessing.OneHotEncoder(handle_unknown="ignore")
        encoder.fit(X_cat)
    X_cat = pd.DataFrame(
        encoder.transform(X_cat).toarray().astype(np.int8),
        columns=encoder.get_feature_names_out(),
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

    return X, scaler, encoder


def classify_lifetime(lifetime) -> Literal[0, 1, 2, 3, 4] | None:
    """Classify lifetime into categories."""
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
    type: str = 'regression',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform train/test split"""
    print("Collecting data...")
    data = open_parquet(cfg.paths.parquets, cfg.files.companies_feat)
    print("Initial data shape:", data.shape)

    data = data.dropna()
    print("Final data shape:", data.shape)

    data.drop(data.loc[data["Основной вид деятельности"] == "na"].index, inplace=True)
    data.drop(data.loc[data["Основной вид деятельности"] == "No"].index, inplace=True)
    if type != 'regression':
        data["lifetime"] = data["lifetime"].apply(classify_lifetime)
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

    # Converting string categories into numerical representations
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
    X_train, scaler, encoder = preprocess_features(X_train)
    X_test, _, _ = preprocess_features(X_test, scaler, encoder)
    save_pickle(cfg.paths.pkls, cfg.files.num_scaler, scaler)
    save_pickle(cfg.paths.pkls, cfg.files.cat_enc, encoder)
    save_pickle(cfg.paths.pkls, cfg.files.column_order, X_train.columns.tolist())
    return X_train, X_test, y_train, y_test


def run_classification(cfg: DictConfig, suffix: str):
    cfg.files.processed_dataset = f"processed_dataset_classification_{suffix}.pkl"
    cfg.files.num_scaler = f"num_scaler_classification_{suffix}.pkl"
    cfg.files.cat_enc = f"cat_enc_classification_{suffix}.pkl"
    cfg.files.metrics = f"metrics_classification_{suffix}.pkl"
    cfg.files.column_order = f"column_order_classification_{suffix}.pkl"
    metrics_dict = {}
    X_train, X_test, y_train, y_test = train_test(cfg, 'classification')
    save_pickle(
        cfg.paths.pkls,
        cfg.files.processed_dataset,
        data=(X_train, X_test, y_train, y_test),
    )
    metrics_dict = catboost_classification(
        cfg, X_train, y_train, X_test, y_test, metrics_dict
    )
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
    save_parquet(cfg.paths.models, cfg.files.metrics, metrics_table)


def run_regression(cfg: DictConfig, suffix: str):
    cfg.files.processed_dataset = f"processed_dataset_regression_{suffix}.pkl"
    cfg.files.num_scaler = f"num_scaler_regression_{suffix}.pkl"
    cfg.files.cat_enc = f"cat_enc_regression_{suffix}.pkl"
    cfg.files.metrics = f"metrics_regression_{suffix}.pkl"
    cfg.files.column_order = f"column_order_regression_{suffix}.pkl"
    metrics_dict = {}
    X_train, X_test, y_train, y_test = train_test(cfg)
    save_pickle(
        cfg.paths.pkls,
        cfg.files.processed_dataset,
        data=(X_train, X_test, y_train, y_test),
    )
    metrics_dict = linear_regression(
        cfg, X_train, y_train, X_test, y_test, metrics_dict
    )
    metrics_dict = catboost_regression(cfg, X_train, y_train, X_test, y_test, metrics_dict)
    metrics_dict = ridge_regression(cfg, X_train, y_train, X_test, y_test, metrics_dict)
    metrics_dict = xgb_regression(cfg, X_train, y_train, X_test, y_test, metrics_dict)
    
    models = []
    frames = []
    for model_id, d in metrics_dict.items():
        models.append(model_id)
        frames.append(pd.DataFrame.from_dict(d, orient="index"))
    metrics_table = pd.concat(frames, keys=models)
    metrics_table = metrics_table.reset_index()
    print(metrics_table)
    save_parquet(cfg.paths.models, cfg.files.metrics, metrics_table)


def update_filename(file_name, suffix):
    base, ext = os.path.splitext(file_name)
    return f"{base}_{suffix}{ext}"


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="training_config.yaml",
)
def run_train(cfg: DictConfig):
    # update file names based on configuration
    if cfg.use_mean_for_region_features:
        suffix = "avg"
    else:
        suffix = "first_year"
    for key in cfg.files:
        if key not in ["processed_dataset", "num_scaler", "cat_enc", "metrics"]:
            cfg.files[key] = update_filename(cfg.files[key], suffix)

    if cfg.get("run_regression", False):
        run_regression(cfg, suffix)
    if cfg.get("run_classification", False):
        run_classification(cfg, suffix)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_train()
