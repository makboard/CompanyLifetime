import logging
import os
from datetime import datetime
from typing import Any, Tuple, List

import pandas as pd
import numpy as np
from omegaconf import DictConfig

from .pickle_manager import open_pickle, save_parquet, open_parquet
from .utils import preprocess_features


def process_row(df) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Processes a DataFrame by dropping unnecessary columns, modifying specific columns,
    and converting string categories to numerical representations.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: Processed DataFrame and series of ogrn values.
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
    ogrns = df["ОГРН"]
    df = df.drop(columns_to_drop, axis=1, errors="ignore")

    # Modify specific columns
    for col in ["Основной вид деятельности", "Регион"]:
        df[col] = df[col].apply(lambda x: x[1:2] if x[0] == "0" else x[:2])

    # Convert categories to numerical representations
    category_mappings = {
        "Тип субъекта": {"Индивидуальный предприниматель": 1},
        "Вновь созданный": {"Да": 1},
        "Наличие лицензий": {"Да": 1},
    }
    for col, mapping in category_mappings.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    df["КатСубМСП"] = df["КатСубМСП"].astype(int)

    return df, ogrns


def predict_classification(row: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes predictions on a processed row using a trained model.

    Parameters:
    row (pd.DataFrame): A preprocessed single-row DataFrame.
    model: The trained model used for predictions.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Predicted class and probabilities.
    """
    return model.predict(row), model.predict_proba(row)


def predict_df_classification(
    df: pd.DataFrame, model, column_order: List[str], scaler, encoder
) -> pd.DataFrame:
    """
    Applies prediction on an entire DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to predict on.
    model: The trained model used for predictions.
    column_order (List[str]): The expected column order for the model.
    scaler: The scaler used for preprocessing.
    encoder: The encoder used for preprocessing.

    Returns:
    pd.DataFrame: DataFrame containing predictions and probabilities.
    """
    try:
        if df.isnull().values.any():
            logging.info("Some features are NaN and will be filled with 0s.")
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
                "ogrn": ogrns,
                "Predicted Class": predictions,
                "Probabilities": [
                    ", ".join([f"{p:.2f}" for p in prob]) for prob in probabilities
                ],
            }
        )
        return results
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return pd.DataFrame()


def load_data(cfg: DictConfig) -> Tuple[pd.DataFrame, Any, Any, List[str]]:
    """
    Loads data, scaler, encoder, and column order from configuration paths.

    Parameters:
    cfg (DictConfig): Configuration object with file paths.

    Returns:
    Tuple[pd.DataFrame, Any, Any, List[str]]: Loaded DataFrame, scaler, encoder, and column order.
    """
    try:
        df = open_parquet(cfg.paths.parquets, cfg.files.companies_feat)
        scaler = open_pickle(cfg.paths.pkls, cfg.files.num_scaler)
        encoder = open_pickle(cfg.paths.pkls, cfg.files.cat_enc)
        column_order = open_pickle(cfg.paths.pkls, cfg.files.column_order)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    return df, scaler, encoder, column_order


def run_classification(cfg: DictConfig) -> None:
    """
    Run the classification analysis based on the configuration. It supports both
    individual ogrn prediction and df predictions over the entire dataset.

    Parameters:
    cfg (DictConfig): Configuration object with model and prediction settings.
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
                ogrn = int(user_input)
                if ogrn in df["ОГРН"].values:
                    row = df[df["ОГРН"] == ogrn]

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
                        f"Prediction for the input {ogrn}: class {predicted_class}, probabilities [{probabilities_str}]"
                    )
                    if cfg.get("save_results", False):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        os.makedirs(cfg.paths.resutls, exist_ok=True)
                        output_file = os.path.join(
                            cfg.paths.resutls, f"output_{timestamp}.txt"
                        )

                        with open(output_file, "a") as file:
                            prediction_output = f"{timestamp} - Prediction for the input {ogrn}: class {predicted_class}, probabilities [{probabilities_str}]\n"
                            file.write(prediction_output)

                        logging.info(f"Prediction saved to {output_file}")

                else:
                    logging.info("OGRN not found in the DataFrame. Please try again.")

            except ValueError:
                logging.info("Invalid input. Please enter a valid OGRN.")
    else:
        logging.info("Collecting predictions for the entire DataFrame.")
        prediction_results = predict_df_classification(
            df, model, column_order, scaler, encoder
        )
        if not prediction_results.empty:
            logging.info(prediction_results.head())
            if cfg.get("save_results", False):
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_file = f"{cfg.files.df_prediction.replace('.parquet', '')}_{timestamp}.parquet"
                os.makedirs(cfg.paths.resutls, exist_ok=True)
                save_parquet(cfg.paths.resutls, output_file, prediction_results)
                logging.info(f"Results saved to {output_file}")
        else:
            logging.error("Failed to generate predictions.")


def run_regression(cfg: DictConfig):
    """
    Placeholder for future implementation of regression analysis.

    Parameters:
    cfg (DictConfig): Configuration object.

    Raises:
    NotImplementedError: Indicates the function is not yet implemented.
    """
    raise NotImplementedError("Regression analysis not implemented yet.")
