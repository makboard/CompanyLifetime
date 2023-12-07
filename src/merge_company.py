import datetime

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from .pickle_manager import open_parquet, save_parquet


def get_date(reg_date: pd.Timestamp, ogrn: int, msp_date: pd.Timestamp) -> pd.Timestamp:
    """Apply reg_date using year calculated from ogrn"""
    if pd.isnull(reg_date) and msp_date == datetime.datetime(2016, 8, 1):
        reg_date = datetime.datetime(int("20" + str(ogrn)[1:3]), 8, 10)
    elif pd.isnull(reg_date) and msp_date != datetime.datetime(2016, 8, 1):
        reg_date = msp_date
    return reg_date


def get_activity(df: pd.DataFrame, INN: int) -> str:
    """Fills empty 'Основной вид деятельности' cell from duplicating row of the same company"""
    df_mini = df[df["ИНН"] == INN].sort_values(
        by=["Основной вид деятельности"], ascending=True
    )
    activity = df_mini["Основной вид деятельности"].values[0]
    if activity in ["No", "na"]:
        return pd.NA
    return activity


def merge_data(cfg: DictConfig) -> None:
    """Merge data from EGRUL and MSP datasets"""
    tqdm.pandas()
    # Open pkl files
    df_egrul = open_parquet(cfg.paths.parquets, cfg.files.file_egrul)
    df_msp = open_parquet(cfg.paths.parquets, cfg.files.file_msp)

    # Apply datetime format for dates columns
    df_msp["Дата включения в реестр"] = pd.to_datetime(
        df_msp["Дата включения в реестр"], dayfirst=True
    )
    df_msp["Дата исключения из реестра"] = pd.to_datetime(
        df_msp["Дата исключения из реестра"], dayfirst=True
    )

    # Merge two datasets by OGRN number
    merged_ogrn = pd.merge(
        left=df_msp, right=df_egrul, how="left", left_on="ОГРН", right_on="ogrn"
    )
    print("merged")
    merged_ogrn["reg_date"] = merged_ogrn.progress_apply(
        lambda row: get_date(
            row["reg_date"], row["ОГРН"], row["Дата включения в реестр"]
        ),
        axis=1,
    )
    merged_ogrn["end_date"] = merged_ogrn.progress_apply(
        lambda row: row["Дата исключения из реестра"]
        if pd.isnull(row["end_date"])
        else row["end_date"],
        axis=1,
    )

    # Reduce unique values in "Основной вид деятельности" by lowering detailes proveded. For example, code 58.11.1 --> 58
    merged_ogrn["Основной вид деятельности"] = list(
        map(lambda item: item[:2], map(str, merged_ogrn["Основной вид деятельности"]))
    )
    # Fill 'na' in "Основной вид деятельности"
    merged_ogrn["Основной вид деятельности"] = merged_ogrn.progress_apply(
        lambda row: get_activity(merged_ogrn, row["ИНН"])
        if (
            (row["Основной вид деятельности"] == "na")
            or (row["Основной вид деятельности"] == "No")
        )
        else row["Основной вид деятельности"],
        axis=1,
    )

    # Calculate lifetime of a company as a separate column
    merged_ogrn["lifetime"] = (
        merged_ogrn["end_date"].dt.year - merged_ogrn["reg_date"].dt.year
    ) * 12 + (merged_ogrn["end_date"].dt.month - merged_ogrn["reg_date"].dt.month)
    # merged_ogrn['lifetime'] = (merged_ogrn['end_date'] - merged_ogrn['reg_date'] ).astype('timedelta64[M]').astype('Int64')

    # Drop duplicates
    merged_ogrn = merged_ogrn.sort_values("reg_date", ascending=False).drop_duplicates(
        subset=["ОГРН"], keep="first"
    )
    if cfg.get("save_open_companies", False):
        companies_open = merged_ogrn[merged_ogrn["Дата исключения из реестра"].isna()]

        # Drop few columns
        companies_open.drop(
            columns=[
                "capital",
                "Дата исключения из реестра",
                "min_num",
                "max_num",
                "end_date",
            ],
            inplace=True,
        )

        save_parquet(cfg.paths.parquets, cfg.files.companies, companies_open)

    else:
        companies_closed = merged_ogrn[
            merged_ogrn["Дата исключения из реестра"].notna()
        ]

        # Drop few columns
        companies_closed.drop(
            columns=[
                "ogrn",
                "opf_id",
                "full_name",
                "okved_id",
                "inn",
                "capital",
                "Дата включения в реестр",
                "Дата исключения из реестра",
                "min_num",
                "max_num",
                "end_date",
            ],
            inplace=True,
        )

        save_parquet(cfg.paths.parquets, cfg.files.companies, companies_closed)


def region_list(cfg: DictConfig) -> None:
    """Extract region codes and coresponding region names"""
    data = open_parquet(cfg.paths.parquets, cfg.files.companies)

    regions = pd.DataFrame(columns=["Name"])

    regions_full = data["Регион"].unique()
    for region_full in regions_full:
        region_full = region_full.replace("г.", "")
        if not any(name == region_full[5:20] for name in list(regions["Name"])):
            regions.loc[region_full[:2], "Name"] = region_full[5:20]
    regions.sort_index(inplace=True)

    # Save dataframe
    save_parquet(cfg.paths.parquets, cfg.files.regions, regions)


if __name__ == "__main__":
    merge_data()
