import numpy as np
import pandas as pd
import xgboost
from catboost import CatBoostRegressor
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .pickle_manager import open_pickle, save_pickle


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


def random_search_grid_cv(regressor, params, X, y) -> dict:
    """
    Perform random search grid cross-validation.

    Args:
    regressor: Regressor instance.
    params: Hyperparameter grid.
    X: Feature data.
    y: Target data.

    Returns:
    Dictionary of best parameters.
    """
    params = OmegaConf.to_container(params, resolve=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=params,
        n_iter=25,
        cv=skf,
        verbose=3,
        random_state=42,
        scoring="f1_macro",
    )
    grid_search.fit(X, y)
    return grid_search.best_params_


def get_predictions(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    paths_models: list,
    predictions_dict: dict,
) -> None:
    for path_model in paths_models:
        regressor = open_pickle(cfg.paths.models, path_model)
        train_predictions = regressor.predict(X_train)
        test_predictions = regressor.predict(X_test)
        key = path_model.split("/")[-1].split(".pkl")[0]
        predictions_dict[key + "_train"] = train_predictions
        predictions_dict[key + "_test"] = test_predictions
    return predictions_dict


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
    default_params = cfg.regressors.linear_regression.default_params
    regressor = LinearRegression(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.regressors.linear_regression.grid_params
        best_params = random_search_grid_cv(regressor, grid_params, X_train, y_train)
        regressor.set_params(**best_params)
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
    default_params = cfg.regressors.ridge_regression.default_params
    regressor = Ridge(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.regressors.ridge_regression.grid_params
        best_params = random_search_grid_cv(regressor, grid_params, X_train, y_train)
        regressor.set_params(**best_params)
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
    default_params = cfg.regressors.xgboost.default_params
    regressor = xgboost.XGBRegressor(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.regressors.xgboost.grid_params
        best_params = random_search_grid_cv(regressor, grid_params, X_train, y_train)
        regressor.set_params(**best_params)
    regressor.fit(X_train, y_train)
    test_predictions = regressor.predict(X_test)
    train_predictions = regressor.predict(X_train)

    save_pickle(cfg.paths.models, cfg.files.xgb_regressor_model, regressor)
    key = "xgb"
    metrics_dict = metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict


def catboost_regression(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Catboost Regression"""
    print("Fitting Catboost Regression...")
    default_params = cfg.regressors.catboost.default_params
    regressor = CatBoostRegressor(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.regressors.catboost.grid_params
        best_params = random_search_grid_cv(regressor, grid_params, X_train, y_train)
        regressor.set_params(**best_params)
    regressor.fit(X_train, y_train)
    train_predictions = regressor.predict(X_train)
    test_predictions = regressor.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.catboost_regressor_model, regressor)
    key = "catboost"
    metrics_dict = metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict
