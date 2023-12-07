import sys, os
import hydra
from omegaconf import DictConfig

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
    if cfg.save_open_companies:
        cfg.files.companies_feat = "companies_feat_open.parquet"
    elif cfg.use_mean_for_region_features:
        cfg.files.companies_feat = "companies_feat_avg.parquet"
    else:
        cfg.files.companies_feat = "companies_feat_first_year.parquet"

    if cfg.get("run_load_data", False):
        load_data(cfg)
    if cfg.get("run_merge_data", False):
        merge_data(cfg)
    if cfg.get("run_region_list", False):
        region_list(cfg)
    if cfg.get("run_add_features", False):
        add_features(cfg)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    run_load()
