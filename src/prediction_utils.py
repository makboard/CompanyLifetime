import logging
import os
from datetime import datetime
from typing import Any, Tuple, List

import pandas as pd
import catboost
import numpy as np
from omegaconf import DictConfig

from .pickle_manager import open_pickle, save_parquet, open_parquet
from .dataset_manager import DatasetManager


def process_row(df) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Processes a DataFrame by modifying specific columns,
    and converting string categories to numerical representations.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: Processed DataFrame and series of ogrn values.
    """
    ogrns = df["ОГРН"]
    # Modify specific columns
    for col in ["Регион"]:  # 'Основной вид деятельности'
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
    df: pd.DataFrame,
    model,
    dataset_manager,
) -> pd.DataFrame:
    """
    Applies prediction on an entire DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to predict on.
    model: The trained model used for predictions.
    dataset_manager: The instance of the DatasetManager class.

    Returns:
    pd.DataFrame: DataFrame containing predictions and probabilities.
    """
    try:
        if df.isnull().values.any():
            logging.info("Some features are NaN and will be filled with 0s.")
            df = df.fillna(0)
        processed_df, ogrns = process_row(df)
        processed_df = dataset_manager.transform_new_data(processed_df)
        # making predictions
        predictions = model.predict(processed_df).squeeze()
        probabilities = model.predict_proba(processed_df).squeeze()
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


def predict_regression(row: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Makes predictions on a processed row using a trained model.

    Parameters:
    row (pd.DataFrame): A preprocessed single-row DataFrame.
    model: The trained model used for predictions.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Predicted value and optionally
        the standard deviation for CatBoostRegressor models.
    """
    if not isinstance(row, pd.DataFrame):
        raise ValueError("Input row must be a pandas DataFrame.")

    if isinstance(model, catboost.core.CatBoostRegressor):
        try:
            predicts = model.virtual_ensembles_predict(
                row,
                prediction_type="TotalUncertainty",
                ntree_end=0,
                virtual_ensembles_count=model._tree_count // 2,
                thread_count=-1,
            )
            mean = predicts[:, 0]
            var = predicts[:, 1]
            return mean, np.sqrt(var)
        except Exception as e:
            raise RuntimeError(f"Error in CatBoostRegressor prediction: {e}")

    elif hasattr(model, "predict"):
        return model.predict(row), None
    else:
        raise TypeError("The model provided is not a recognized type.")


def predict_df_regression(df: pd.DataFrame, model, dataset_manager) -> pd.DataFrame:
    """
    Applies prediction on an entire DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to predict on.
    model: The trained model used for predictions.
    dataset_manager: The instance of the DatasetManager class.

    Returns:
    pd.DataFrame: DataFrame containing the predicted lifetime and optionally
        the standard deviation for CatBoostRegressor models.
    """
    try:
        if df.isnull().values.any():
            logging.info("Some features are NaN and will be filled with 0s.")
            df = df.fillna(0)
        processed_df, ogrns = process_row(df)
        processed_df = dataset_manager.transform_new_data(processed_df)
        # Making predictions
        if isinstance(model, catboost.core.CatBoostRegressor):
            predicts = model.virtual_ensembles_predict(
                processed_df,
                prediction_type="TotalUncertainty",
                ntree_end=0,
                virtual_ensembles_count=model._tree_count // 2,
                thread_count=-1,
            )
            lifetime = predicts[:, 0]
            std_dev = np.sqrt(predicts[:, 1])
        else:
            lifetime = model.predict(processed_df)
            std_dev = None

        results = pd.DataFrame(
            {
                "ogrn": ogrns,
                "Predicted Lifetime": lifetime,
                "Standard Deviation": std_dev,
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
        dataset_manager = DatasetManager.load_instance(
            os.path.join(cfg.paths.pkls, cfg.files.data_manager)
        )
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    return df, dataset_manager


def run_classification(cfg: DictConfig) -> None:
    """
    Run the classification analysis based on the configuration. It supports both
    individual ogrn prediction and df predictions over the entire dataset.

    Parameters:
    cfg (DictConfig): Configuration object with model and prediction settings.
    """
    logging.info("Collecting data...")
    try:
        df, dataset_manager = load_data(cfg)
        model = open_pickle(cfg.paths.models, cfg.files.classification_model)
    except Exception as e:
        logging.error(f"Error in data loading or model opening: {e}")
        return
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
                    processed_row = process_row(row)[0]
                    processed_row = dataset_manager.transform_new_data(processed_row)

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
        prediction_results = predict_df_classification(df, model, dataset_manager)
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
    Run the regression analysis based on the configuration. It supports both
    individual ogrn prediction and df predictions over the entire dataset.

    Parameters:
    cfg (DictConfig): Configuration object with model and prediction settings.
    """
    logging.info("Collecting data...")
    try:
        df, dataset_manager = load_data(cfg)
        model = open_pickle(cfg.paths.models, cfg.files.regression_model)
    except Exception as e:
        logging.error(f"Error in data loading or model opening: {e}")
        return

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
                    processed_row = process_row(row)[0]
                    processed_row = dataset_manager.transform_new_data(processed_row)

                    # predict
                    prediction, std = predict_regression(processed_row, model)
                    if std is not None:
                        logging.info(
                            f"Prediction for the input {ogrn}: lifetime {prediction} +- {3 * std}"
                        )
                    else:
                        logging.info(
                            f"Prediction for the input {ogrn}: lifetime {prediction}"
                        )
                    if cfg.get("save_results", False):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        os.makedirs(cfg.paths.resutls, exist_ok=True)
                        output_file = os.path.join(
                            cfg.paths.resutls, f"output_{timestamp}.txt"
                        )

                        with open(output_file, "a") as file:
                            prediction_output = f"{timestamp} - Prediction for the input {ogrn}: lifetime {prediction}, standard deviation {std}\n"
                            file.write(prediction_output)

                        logging.info(f"Prediction saved to {output_file}")

                else:
                    logging.info("OGRN not found in the DataFrame. Please try again.")

            except ValueError:
                logging.info("Invalid input. Please enter a valid OGRN.")
    else:
        logging.info("Collecting predictions for the entire DataFrame.")
        try:
            prediction_results = predict_df_regression(df, model, dataset_manager)
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
        except Exception as e:
            logging.error(f"Error during batch prediction: {e}")
