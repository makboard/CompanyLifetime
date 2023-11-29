import pandas as pd

import os
import sys

import hydra
from omegaconf import DictConfig

import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.getcwd())
from src.utils import update_filename, update_config_filenames
from src.pickle_manager import save_pickle, open_pickle, open_parquet, save_parquet
from src.regression_models import (
    linear_regression,
    ridge_regression,
    xgb_regression,
    differentiated_metrics,
    get_predictions,
)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="training_config.yaml",
)
def run_test(cfg: DictConfig):
    metrics_dict = {}
    predictions_dict = {}
    suffix = "regression"
    for key in cfg.files:
        if key not in ["processed_dataset", "num_scaler", "cat_enc", "metrics"]:
            cfg.files[key] = update_filename(cfg.files[key], suffix)
    cfg = update_config_filenames(cfg, "classification", suffix)
    X_train, X_test, y_train, y_test = open_pickle(cfg.files, cfg.processed_dataset)
    paths_models = [
        cfg.files.lr_model,
        cfg.files.ridge_model,
        cfg.files.xgb_regressor_model,
    ]
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
