from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
import xgboost
import numpy as np
import pandas as pd

from omegaconf import DictConfig
import sys
import os

sys.path.append(os.getcwd())
from src.pickle_manager import save_pickle, open_pickle


def metrics_print(
    y_train: pd.DataFrame,
    train_predictions: np.ndarray,
    y_test: pd.DataFrame,
    test_predictions: np.ndarray,
    metrics_dict: dict,
    key: str,
) -> dict:
    """Print regression metrics on train amd test data"""
    r2_train = metrics.r2_score(y_train, train_predictions)
    mae_train = metrics.mean_absolute_error(y_train, train_predictions)
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_predictions))
    print("Train \nr2 score:", r2_train)
    print("MAE:", mae_train)
    print("RMSE:", rmse_train)

    r2_test = metrics.r2_score(y_test, test_predictions)
    mae_test = metrics.mean_absolute_error(y_test, test_predictions)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, test_predictions))
    print("Test \nr2 score:", r2_test)
    print("MAE:", mae_test)
    print("RMSE:", rmse_test)

    metrics_dict[key] = {"train": {}, "test": {}}
    (
        metrics_dict[key]["train"]["r2"],
        metrics_dict[key]["train"]["mae"],
        metrics_dict[key]["train"]["rmse"],
    ) = (r2_train, mae_train, rmse_train)
    (
        metrics_dict[key]["test"]["r2"],
        metrics_dict[key]["test"]["mae"],
        metrics_dict[key]["test"]["rmse"],
    ) = (r2_test, mae_test, rmse_test)

    return metrics_dict


def differentiated_metrics(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    paths_models: list,
    metrics_dict: dict,
) -> None:
    for path_model in paths_models:
        regressor = open_pickle(cfg.paths.models, path_model)
        train_predictions = regressor.predict(X_train)
        test_predictions = regressor.predict(X_test)
        key = path_model.split("/")[-1].split(".pkl")[0]
        target_categories = cfg.test.categories
        for i in range(1, len(target_categories)):
            tmin, tmax = target_categories[i - 1], target_categories[i]
            idxs = np.where((y_test >= tmin) & (y_test < tmax))[0]
            metrics_dict = metrics_print(
                y_train[idxs],
                train_predictions[idxs],
                y_test[idxs],
                test_predictions[idxs],
                metrics_dict,
                key + "_" + "_".join([str(tmin), str(tmax)]),
            )


def linear_regression(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit Linear Regression"""
    print("Fitting linear regression...")
    regressor = LinearRegression(**cfg.linear_regression)
    regressor.fit(X_train, y_train)
    train_predictions = regressor.predict(X_train)
    test_predictions = regressor.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.lr_model, regressor)
    key = "lr"
    metrics_dict = metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict


def ridge_regression(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit Ridge Regression"""
    print("Fitting ridge regression...")
    regressor = Ridge(**cfg.ridge_regression)
    regressor.fit(X_train, y_train)
    train_predictions = regressor.predict(X_train)
    test_predictions = regressor.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.ridge_model, regressor)
    key = "ridge"
    metrics_dict = metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict


def xgb_regression(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit XGBoostRegressor"""
    print("Fitting XGB regression...")
    regressor = xgboost.XGBRegressor(**cfg.xgboost)

    regressor.fit(X_train, y_train)
    test_predictions = regressor.predict(X_test)
    train_predictions = regressor.predict(X_train)

    save_pickle(cfg.paths.models, cfg.files.xgb_model, regressor)
    key = "xgb"
    metrics_dict = metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict
