from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import xgboost
import numpy as np
import pandas as pd

from omegaconf import OmegaConf, DictConfig
import sys
import os

sys.path.append(os.getcwd())
from src.pickle_manager import save_pickle


def classification_metrics_print(
    y_train: pd.DataFrame,
    train_predictions: np.ndarray,
    y_test: pd.DataFrame,
    test_predictions: np.ndarray,
    metrics_dict: dict,
    key: str,
) -> dict:
    """Print classification metrics on train and test data"""
    accuracy_train = metrics.accuracy_score(y_train, train_predictions)
    precision_train = metrics.precision_score(
        y_train, train_predictions, average="weighted"
    )
    recall_train = metrics.recall_score(y_train, train_predictions, average="weighted")
    f1_train = metrics.f1_score(y_train, train_predictions, average="weighted")
    print("Train \nAccuracy:", accuracy_train)
    print("Precision:", precision_train)
    print("Recall:", recall_train)
    print("F1 Score:", f1_train)

    accuracy_test = metrics.accuracy_score(y_test, test_predictions)
    precision_test = metrics.precision_score(
        y_test, test_predictions, average="weighted"
    )
    recall_test = metrics.recall_score(y_test, test_predictions, average="weighted")
    f1_test = metrics.f1_score(y_test, test_predictions, average="weighted")
    print("Test \nAccuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)
    print("F1 Score:", f1_test)

    metrics_dict[key] = {"train": {}, "test": {}}
    metrics_dict[key]["train"]["accuracy"] = accuracy_train
    metrics_dict[key]["train"]["precision"] = precision_train
    metrics_dict[key]["train"]["recall"] = recall_train
    metrics_dict[key]["train"]["f1"] = f1_train

    metrics_dict[key]["test"]["accuracy"] = accuracy_test
    metrics_dict[key]["test"]["precision"] = precision_test
    metrics_dict[key]["test"]["recall"] = recall_test
    metrics_dict[key]["test"]["f1"] = f1_test

    return metrics_dict


def random_search_grid_cv(classifier, params, X, y) -> dict:
    """
    Perform random search grid cross-validation.

    Args:
    classifier: Classifier instance.
    params: Hyperparameter grid.
    X: Feature data.
    y: Target data.

    Returns:
    Dictionary of best parameters.
    """
    params = OmegaConf.to_container(params, resolve=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=params,
        n_iter=25,
        cv=skf,
        verbose=3,
        random_state=42,
        scoring="f1_macro",
    )
    grid_search.fit(X, y)
    return grid_search.best_params_


def logistic_regression_classification(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit Logistic Regression for classification"""
    print("Fitting Logistic Regression...")
    default_params = cfg.classifiers.log_reg.default_params
    classifier = LogisticRegression(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.classifiers.log_reg.grid_params
        best_params = random_search_grid_cv(classifier, grid_params, X_train, y_train)
        classifier.set_params(**best_params)
    classifier.fit(X_train, y_train)
    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.log_reg_model, classifier)
    key = "logistic_regression"
    metrics_dict = classification_metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict


def random_forest_classification(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit Random Forest Classifier for classification"""
    print("Fitting Random Forest Classifier...")
    default_params = cfg.classifiers.rf.default_params
    classifier = RandomForestClassifier(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.classifiers.rf.grid_params
        best_params = random_search_grid_cv(classifier, grid_params, X_train, y_train)
        classifier.set_params(**best_params)
    classifier.fit(X_train, y_train)
    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.rf_classifier_model, classifier)
    key = "random_forest"
    metrics_dict = classification_metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict


def xgb_classification(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics_dict: dict,
) -> dict:
    """Fit XGBoostClassifier for classification"""
    print("Fitting XGBoost Classifier...")
    default_params = cfg.classifiers.xgboost.default_params
    classifier = xgboost.XGBClassifier(**default_params)
    if cfg.enable_optimization:
        grid_params = cfg.classifiers.xgboost.grid_params
        best_params = random_search_grid_cv(classifier, grid_params, X_train, y_train)
        classifier.set_params(**best_params)
    classifier.fit(X_train, y_train)
    train_predictions = classifier.predict(X_train)
    test_predictions = classifier.predict(X_test)

    save_pickle(cfg.paths.models, cfg.files.xgb_classifier_model, classifier)
    key = "xgb"
    metrics_dict = classification_metrics_print(
        y_train, train_predictions, y_test, test_predictions, metrics_dict, key
    )
    return metrics_dict
