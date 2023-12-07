from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import make_indices


class PassthroughTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.feature_names: Optional[List[str]] = None

    def fit(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PassthroughTransformer":
        self.feature_names = df.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        return self.feature_names


class DatasetManager:
    def __init__(self):
        self.preprocessor: Optional[ColumnTransformer] = None
        self.column_order: Optional[List[str]] = None
        self.classification: Optional[bool] = None
        self.original_columns: Optional[List[str]] = None
        
    def make_indices(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits feature indices into binary, numerical, and categorical.
        Parameters:
        df (pd.DataFrame): DataFrame with features.
        Returns:
        Tuple[List[str], List[str], List[str]]: Lists of binary, categorical,
            and numerical column names.
        """
        binary_columns = ["Тип субъекта", "Вновь созданный", "Наличие лицензий"]
        categorical_columns = ["Основной вид деятельности", "Регион", "КатСубМСП"]
        binary_indices = df.columns.isin(binary_columns)
        categorical_indices = df.columns.isin(categorical_columns)
        numerical_columns = df.columns[~(binary_indices | categorical_indices)]

        return binary_columns, categorical_columns, numerical_columns.tolist()

    def create_preprocessor(self, df: pd.DataFrame) -> None:
        """
        Creates a preprocessor pipeline for the given DataFrame.

        Parameters:
        df (pd.DataFrame): DataFrame used to determine the preprocessing strategy.
        """
        binary_columns, categorical_columns, numerical_columns = make_indices(df)

        num_pipeline = Pipeline([("scaler", preprocessing.StandardScaler())])

        cat_pipeline = Pipeline(
            [("encoder", preprocessing.OneHotEncoder(handle_unknown="ignore"))]
        )

        bin_pipeline = Pipeline([("passthrough", PassthroughTransformer())])

        self.preprocessor = ColumnTransformer(
            [
                ("num", num_pipeline, numerical_columns),
                ("cat", cat_pipeline, categorical_columns),
                ("bin", bin_pipeline, binary_columns),
            ]
        )

    def classify_lifetime(self, y: pd.Series) -> pd.Series:
        """
        Classifies lifetime into categories using vectorized operations.

        Parameters:
        y (pd.Series): Series of lifetime values to classify.

        Returns:
        pd.Series: Series of classified categories.
        """
        categories = pd.Series(0, index=y.index)  # Default category
        categories[(y > 24) & (y <= 120)] = 1
        categories[y > 120] = 2
        return categories

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 123,
        classification: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Fits and transforms the dataset, splitting it into training and testing sets.

        Parameters:
        df (pd.DataFrame): The feature dataset.
        y (pd.Series): The target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        classification (bool): Determines if the target variable should be treated as 
            a classification problem.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: The transformed training and testing 
            datasets, along with their respective target variables.
        """
        if classification:
            self.classification = True
            y = self.classify_lifetime(y)
            stratify = y
        else:
            self.classification = False
            stratify = None
        # Store original column names
        self.original_columns = df.columns.tolist()

        x_train, x_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.create_preprocessor(x_train)
        self.preprocessor.fit(x_train)
        # Transform the data
        transformed_x_train = self.preprocessor.transform(x_train)
        transformed_x_test = self.preprocessor.transform(x_test)

        # Convert to dense matrix if output is sparse
        if scipy.sparse.issparse(transformed_x_train):
            transformed_x_train = transformed_x_train.toarray()
        if scipy.sparse.issparse(transformed_x_test):
            transformed_x_test = transformed_x_test.toarray()

        # Create DataFrame with consistent column names
        feature_names = self.get_feature_names()
        x_train_preprocessed = pd.DataFrame(transformed_x_train, columns=feature_names)
        x_test_preprocessed = pd.DataFrame(transformed_x_test, columns=feature_names)

        self.column_order = x_train_preprocessed.columns.tolist()

        return x_train_preprocessed, x_test_preprocessed, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """
        Extracts feature names from the preprocessor.

        Returns:
        List[str]: The concatenated feature names from the preprocessor.
        """
        num_features = self.preprocessor.named_transformers_["num"][
            "scaler"
        ].get_feature_names_out()
        cat_features = self.preprocessor.named_transformers_["cat"][
            "encoder"
        ].get_feature_names_out()
        bin_features = self.preprocessor.named_transformers_[
            "bin"
        ].get_feature_names_out()

        return np.concatenate([num_features, cat_features, bin_features]).tolist()

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using the fitted preprocessor.

        Parameters:
        df (pd.DataFrame): New data to transform.

        Returns:
        pd.DataFrame: The transformed DataFrame.
        """
        # Check if df contains all required original columns
        missing_columns = set(self.original_columns) - set(df.columns)
        assert (
            missing_columns == set()
        ), f"DataFrame is missing required original columns: {missing_columns}"

        transformed_df = self.preprocessor.transform(df)

        if scipy.sparse.issparse(transformed_df):
            transformed_df = transformed_df.toarray()

        df_preprocessed = pd.DataFrame(transformed_df, columns=self.column_order)

        return df_preprocessed

    def save_instance(self, path: str) -> None:
        """
        Saves the entire class instance to a file.

        Parameters:
        path (str): File path for saving the instance.
        """
        joblib.dump(self, path)

    @staticmethod
    def load_instance(path: str) -> "DatasetManager":
        """
        Loads the class instance from a file.

        Parameters:
        path (str): File path for loading the instance.

        Returns:
        DatasetManager: The loaded class instance.
        """
        return joblib.load(path)
