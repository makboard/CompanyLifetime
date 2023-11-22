from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os
import sys

import hydra
from omegaconf import DictConfig

import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
from src.pickle_manager import save_pickle, open_pickle, open_parquet, save_parquet
from src.regression_models import (
    linear_regression,
    ridge_regression,
    xgb_regression,
    differentiated_metrics,
    get_predictions,
)


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


def preprocess_X(X: pd.DataFrame) -> pd.DataFrame:
    """Perform scaling and encoding for particular columns"""
    binary_columns, categorical_columns, numerical_columns = make_indices(X)

    # Scale numerical
    scaler = preprocessing.StandardScaler(with_mean=0)
    X_num = X[numerical_columns]
    X_num = scaler.fit_transform(X_num)
    X_num = pd.DataFrame(X_num, columns=numerical_columns)

    # Encode categorical
    enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
    X_cat = X[categorical_columns]
    X_cat = enc.fit_transform(X_cat)
    X_cat = pd.DataFrame(
        X_cat.toarray().astype(np.int8), columns=enc.get_feature_names_out()
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

    return X


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

    X = preprocess_X(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    return X_train, X_test, y_train, y_test


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="regression_config.yaml",
)
def run_test(cfg: DictConfig):
    metrics_dict = {}
    predictions_dict = {}
    X_train, X_test, y_train, y_test = train_test(cfg)
    paths_models = ["lr.pkl", "ridge.pkl", "xgb.pkl"]
    get_predictions(
        cfg, X_train, y_train, X_test, y_test, paths_models, predictions_dict
    )
    differentiated_metrics(
        cfg, X_train, y_train, X_test, y_test, paths_models, metrics_dict
    )
    # metrics_dict = linear_regression(
    # cfg, X_train, y_train, X_test, y_test, metrics_dict
    # )
    # metrics_dict = ridge_regression(cfg, X_train, y_train, X_test, y_test, metrics_dict)
    # metrics_dict = xgb_regression(cfg, X_train, y_train, X_test, y_test, metrics_dict)
    models = []
    frames = []

    for model_id, d in metrics_dict.items():
        models.append(model_id)
        frames.append(pd.DataFrame.from_dict(d, orient="index"))
    metrics_table = pd.concat(frames, keys=models)
    metrics_table = metrics_table.reset_index()
    print(metrics_table)
    save_parquet(cfg.paths.models, cfg.files.metrics, metrics_table)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_test()
