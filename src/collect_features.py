import os
import sys
import pandas as pd
import numpy as np

from omegaconf import DictConfig
import xml.etree.ElementTree as ET
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.pickle_manager import open_pickle, save_pickle, open_parquet, save_parquet


def collect_regional(cfg: DictConfig) -> tuple:
    """Collect regional features"""
    print("Collecting regional features...")
    regions = open_parquet(cfg.paths.parquets, cfg.files.regions)
    years = np.arange(cfg.foundation_start, cfg.foundation_end)
    years = [str(year) for year in years]
    n_years = len(years)

    # Load list of features tags to use
    xl = pd.ExcelFile(os.path.join(cfg.paths.data, cfg.files.features))
    tags = (xl.parse().iloc[:, 0]).tolist()

    # Choose the files with data
    filenames = [
        file for file in os.listdir(cfg.paths.features_region) if "Раздел" in file
    ]
    os.makedirs("../" + cfg.paths.data, exist_ok=True)

    feat_array = np.empty((regions.shape[0] - 2, n_years, 0))
    tags_order = []

    for file in tqdm(filenames):
        # Load excel file
        xl = pd.ExcelFile(os.path.join(cfg.paths.features_region, file))
        sheet_names = xl.sheet_names

        # Filter features
        features = sorted(set(sheet_names) & set(tags), key=sheet_names.index)

        if len(features) != 0:
            for f in features:
                # Calculate anchor row to find skiprows param
                data_ref = xl.parse(sheet_name=f).iloc[:, 0]
                skip_value = data_ref.str.contains("Белгородская область")
                skip_value.fillna(False)
                index = np.where(skip_value)[0][0]

                data = xl.parse(sheet_name=f, skiprows=index + 1)

                # Skip all superscripts in years
                data.columns = [str(name)[:4] for name in data.columns]

                # Leave required regional lines only
                for row in data.itertuples():
                    if not any(
                        reg_name in row[1] for reg_name in regions["Name"].tolist()
                    ):
                        data = data.drop(index=row.Index)

                # Add years not shown in data, put Nans there
                add_cols = list(set(years) ^ set(data.columns[1:]))
                for col in add_cols:
                    data[col] = np.nan

                # Sort years and replace weird symbols with 0
                data = data.sort_index(axis=1)
                data_cols = data.columns[:-1]
                data[data_cols] = data[data_cols].replace(
                    to_replace=["-", ".*"], value=0, regex=True
                )

                tags_order.append(f)
                data1 = data.iloc[:, :n_years].fillna(0)
                feat_array = np.dstack((feat_array, data1))

    # Regional codes in data order
    codes = []
    for r in data.iloc[:, -1]:
        if "г." not in r:
            index = regions.index[regions["Name"] == r[:15]].tolist()[0]
        else:
            index = regions.index[regions["Name"] == r[3:18]].tolist()[0]
        codes.append(index)

    return feat_array, tags_order, codes


def apply_regional(
    row: pd.Series, region_features: list, tags_order: list, codes: list
) -> pd.Series:
    "Makes a slice of regional features for particular sample"
    # global region_features, tags_order, codes
    # Years to col indexes
    reg_year = row["reg_date"].year - 2000 - 1  # year previous to registration

    # Assign for samples with late reg_year the smaller value (Rosstat data'2021 is the last available)
    if reg_year >= 22:
        reg_year = 21
    closed_year = reg_year + int(row["lifetime"] // 12) + 1
    # Find region line and corresponding index
    region = row["Регион"][:2]

    # Skip ex-Ukrainian regions with sparse data
    if region in ["90", "93", "94", "95"]:
        array_extract = np.full([region_features.shape[2]], np.nan)

    # For any other region go on
    else:
        region_line = [code == region for code in codes]
        region_index = np.where(region_line)[0][0]
        array_extract = region_features[region_index, reg_year:closed_year, :].mean(0)
        # array_extract = region_features[region_index, reg_year, :]
        array_extract[array_extract == 0] = np.nan
    row_add = pd.Series(array_extract, index=tags_order)
    row = pd.concat([row, row_add])

    return row


def collect_extra_features(cfg: DictConfig) -> pd.DataFrame:
    """Collect "ССЧР" and "КатСубМСП" features"""
    folders = [
        folder for folder in os.listdir(cfg.paths.features_extra) if "zip" not in folder
    ]
    print("Collecting extra features from {} folders...".format(len(folders)))

    # Put 2022 xml in the beginning (they will remain after removing duplicates)
    for f in folders:
        if "22" in f:
            folders.remove(f)
            folders.insert(0, f)
    # Create an empty list to store the parsed XML data
    xml_data = []

    for folder in folders:
        path_folder = os.path.join(cfg.paths.features_extra, folder)

        # Loop through each file in the folder
        for filename in tqdm(os.listdir(path_folder)):
            # Check if the file is an XML file
            if filename.endswith(".xml"):
                # Parse the XML data from the file
                tree = ET.parse(
                    os.path.join(cfg.paths.features_extra, folder, filename)
                )
                root = tree.getroot()

                # # Loop through each child element in the root element
                for child in root[1:]:
                    # Extract the data from each child element and append it to the xml_data list
                    for subchild in child[:1]:
                        row = {}
                        inn = list(
                            map(
                                subchild.attrib.get,
                                filter(
                                    lambda x: x in ["ИННЮЛ", "ИННФЛ"], subchild.attrib
                                ),
                            )
                        )

                        row["ИНН"] = inn[0]
                        row["КатСубМСП"] = child.attrib["КатСубМСП"]
                        row["ССЧР"] = child.get(
                            "ССЧР"
                        )  # puts None if key doesn't exist
                    xml_data.append(row)

    # Convert the xml_data list to DataFrame
    extra_feat = pd.DataFrame(xml_data)

    # Fill with ones (these samples are ИП lacking this feature)
    extra_feat["ССЧР"].fillna(value=1, inplace=True)

    # Drop duplicates and weird samples
    extra_feat = extra_feat.drop_duplicates(
        subset=["ИНН"],
        keep="first",
    )
    bad_index = extra_feat[extra_feat["ИНН"] == "~~~~~~~~~~~~"]
    if not bad_index.empty:
        extra_feat = extra_feat.drop(index=[bad_index.index[0]])
    extra_feat = extra_feat.astype(
        {"ИНН": "int64", "ССЧР": "int32", "КатСубМСП": "int8"}
    )
    save_parquet(cfg.paths.parquets, cfg.files.extra_features, extra_feat)

    return extra_feat


def add_features(cfg: DictConfig) -> None:
    """Put features to companies DataFrame"""
    tqdm.pandas()
    region_features, tags_order, codes = collect_regional(cfg)

    # Load features from .xml files
    # extra_features = collect_extra_features(cfg)
    # Load same features from existing .parquet file
    extra_features = open_parquet(cfg.paths.parquets, cfg.files.extra_features)

    print("Stacking all features...")
    df = open_parquet(cfg.paths.parquets, cfg.files.companies)
    df = df.progress_apply(
        lambda row: apply_regional(row, region_features, tags_order, codes), axis=1
    )

    # Extra features
    result = df.merge(extra_features, on="ИНН", how="left")

    # Save
    save_parquet(cfg.paths.parquets, cfg.files.companies_feat, result)


# def merge_features(cfg:DictConfig):
#     '''Collect and add regional and additional features'''
#     add_features(cfg)


if __name__ == "__main__":
    # add_features(cfg)
    pass
