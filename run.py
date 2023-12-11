import logging
import os
import sys

import hydra
from omegaconf import DictConfig

from src.utils import update_filename
from src.prediction_utils import run_classification, run_regression

logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="inference_config.yaml",
)
def main(cfg: DictConfig):
    if cfg.get("run_regression", False):
        cfg.files.data_manager = update_filename(
            cfg.files.data_manager, "regression_first_year"
        )
        run_regression(cfg)
    if cfg.get("run_classification", False):
        cfg.files.data_manager = update_filename(
            cfg.files.data_manager, "classification_first_year"
        )
        run_classification(cfg)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
