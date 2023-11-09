import sys, os
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig

sys.path.append(os.getcwd())
from src.load_company import load_data
from src.merge_company import merge_data, region_list
from src.collect_features import add_features


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="dataset_config.yaml",
)
def run_load(cfg: DictConfig):
    load_data(cfg)
    merge_data(cfg)
    region_list(cfg)
    add_features(cfg)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_load()
