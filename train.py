import os
import sys
import warnings

import hydra
from omegaconf import DictConfig

from src.utils import update_filename, run_classification, run_regression

# Configure warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())


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
        if key not in ["processed_dataset", "data_manager", "metrics"]:
            cfg.files[key] = update_filename(cfg.files[key], suffix)

    if cfg.get("run_regression", False):
        run_regression(cfg, suffix)
    if cfg.get("run_classification", False):
        run_classification(cfg, suffix)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_train()
