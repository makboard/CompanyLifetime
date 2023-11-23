import logging
import os
from datetime import datetime
import sys

import pandas as pd
import hydra
from omegaconf import DictConfig

from src.pickle_manager import open_pickle, open_parquet, save_parquet
from train import preprocess_features

logging.basicConfig(level=logging.INFO)


def process_row(df):
    """
    Processes the DataFrame by dropping unnecessary columns, modifying specific columns,
    and converting string categories to numerical representations.
    """
    columns_to_drop = [
        "Наименование / ФИО",
        "Дата включения в реестр",
        "ОГРН",
        "ИНН",
        "reg_date",
        "lifetime",
        "ogrn",
        "opf_id",
        "okved_id",
        "inn",
        "full_name",
    ]
    if sum(df["Основной вид деятельности"] == "No") > 0:
        logging.info(
            "Some companies have not expected type of activity, dropping them..."
        )
        df.drop(df.loc[df["Основной вид деятельности"] == "No"].index, inplace=True)
    ogrns = df["ОГРН"]
    df = df.drop(columns_to_drop, axis=1, errors="ignore")

    df["Основной вид деятельности"] = df.apply(
        lambda row: row["Основной вид деятельности"][1:2]
        if row["Основной вид деятельности"][0] == "0"
        else row["Основной вид деятельности"][:2],
        axis=1,
    )
    df["Регион"] = df.apply(
        lambda row: row["Регион"][1:2]
        if row["Регион"][0] == "0"
        else row["Регион"][:2],
        axis=1,
    )

    df["Тип субъекта"] = (
        df["Тип субъекта"] == "Индивидуальный предприниматель"
    ).astype(int)
    df[["Основной вид деятельности", "Регион"]] = df[
        ["Основной вид деятельности", "Регион"]
    ].astype(int)
    df["Вновь созданный"] = (df["Вновь созданный"] == "Да").astype(int)
    df["Наличие лицензий"] = (df["Наличие лицензий"] == "Да").astype(int)
    df["КатСубМСП"] = df["КатСубМСП"].astype(int)

    return df, ogrns


def predict_classification(row, model):
    """
    Makes predictions on the processed row using the trained model.
    """
    return model.predict(row), model.predict_proba(row)


def predict_batch_classification(df, model, column_order, scaler, encoder):
    """
    Applies prediction on a batch (entire DataFrame).
    """
    try:
        if df.isnull().values.any():
            logging.info("Some features are NaN and will be filled with 0.")
            df = df.fillna(0)
        processed_df, ogrns = process_row(df)
        processed_df = preprocess_features(processed_df, scaler, encoder)[0]
        missing_columns = set(column_order) - set(processed_df.columns)
        if missing_columns:
            logging.info(f"Missing features columns: {missing_columns}")
            return pd.DataFrame()

        predictions = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)
        results = pd.DataFrame(
            {
                "OGRN": ogrns,
                "Predicted Class": predictions,
                "Probabilities": [
                    ", ".join([f"{p:.2f}" for p in prob]) for prob in probabilities
                ],
            }
        )
        return results
    except Exception as e:
        logging.info(f"An error occurred during batch prediction: {e}")
        return pd.DataFrame()


def load_data(cfg: DictConfig):
    """
    Load data, scaler, encoder, and column order from specified file paths in the configuration.
    """
    try:
        df = open_parquet(cfg.paths.parquets, cfg.files.companies_feat):
        df = open_parquet(cfg.paths.parquets, cfg.files.companies_feat)
        scaler = open_pickle(cfg.paths.pkls, cfg.files.num_scaler)
        encoder = open_pickle(cfg.paths.pkls, cfg.files.cat_enc)
        column_order = open_pickle(cfg.paths.pkls, cfg.files.column_order)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise 
    
    return df, scaler, encoder, column_order

def run_classification(cfg: DictConfig):
    """
    Run the classification analysis based on the configuration. It supports both 
    individual OGRN prediction and batch predictions over the entire dataset.
    """
    logging.info("Collecting data...")
    df, scaler, encoder, column_order = load_data(cfg)
    model = open_pickle(cfg.paths.models, cfg.files.classification_model)
    if cfg.predict_by_ogrn:
        while True:
            user_input = input(
                "Enter an OGRN(IP) to find in the processed Data or 'exit' to quit: "
            )

            if user_input.lower() == "exit":
                logging.info("Exiting the program.")
                break

            try:
                OGRN = int(user_input)
                if OGRN in df["ОГРН"].values:
                    row = df[df["ОГРН"] == OGRN]

                    # Check for NaN values and missing columns
                    if row.isnull().values.any():
                        logging.info("Some features are NaN and will be filled with 0.")
                        row = row.fillna(0)

                    # Preprocess row
                    try:
                        row = process_row(row)[0]
                        row = preprocess_features(row, scaler, encoder)[0]
                    except Exception as e:
                        logging.info(f"An error occurred during processing: {e}")
                    missing_columns = set(column_order) - set(row.columns)
                    if missing_columns:
                        logging.info(f"Missing features columns: {missing_columns}")
                        break
                    processed_row = row[column_order]
                    # predict
                    prediction = predict_classification(processed_row, model)
                    predicted_class = prediction[0][0]
                    probabilities_str = ", ".join(
                        [f"{p:.2f}" for p in prediction[1][0]]
                    )
                    logging.info(
                        f"Prediction for the input {OGRN}: class {predicted_class}, probabilities [{probabilities_str}]"
                    )
                    if cfg.get("save_results", False):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        os.makedirs(cfg.paths.resutls, exist_ok=True)
                        output_file = os.path.join(
                            cfg.paths.resutls, f"output_{timestamp}.txt"
                        )

                        with open(output_file, "a") as file:
                            prediction_output = f"{timestamp} - Prediction for the input {OGRN}: class {predicted_class}, probabilities [{probabilities_str}]\n"
                            file.write(prediction_output)

                        logging.info(f"Prediction saved to {output_file}")

                else:
                    logging.info("OGRN not found in the DataFrame. Please try again.")

            except ValueError:
                logging.info("Invalid input. Please enter a valid OGRN.")
    else:
        logging.info("Collecting predictions for the entire DataFrame.")
        prediction_results = predict_batch_classification(
            df, model, column_order, scaler, encoder
        )
        if not prediction_results.empty:
            logging.info(prediction_results.head())
            if cfg.get("save_results", False):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_file = f"{cfg.files.files.df_prediction}_{timestamp}.pkl"
                os.makedirs(cfg.paths.resutls, exist_ok=True)
                save_parquet(cfg.paths.resutls, output_file, prediction_results)
                logging.info(f"Results saved to {output_file}")
        else:
            logging.error("Failed to generate predictions.")


def run_regression(cfg: DictConfig):
    """
    Placeholder function for running regression analysis.
    To be implemented in the future.
    """
    df, scaler, encoder, column_order = load_data(cfg)
    raise NotImplementedError("The run_regression function is not implemented yet.")


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config"),
    config_name="inference_config.yaml",
)
def main(cfg: DictConfig):
    if cfg.get("run_regression", False):
        run_regression(cfg)
    if cfg.get("run_classification", False):
        run_classification(cfg)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
