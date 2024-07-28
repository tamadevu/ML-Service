import pickle
from uuid import uuid4
from fastapi import HTTPException
from numpy import ndarray
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from pydantic import BaseModel, Field
import tempfile
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from ml_service.schemas.regressor import (
    PerformanceMetrics,
    TestModelResponse,
    TrainModelResponse,
)


class RandomForestRegressor(BaseModel):
    target: str = Field(..., description="Target column name")
    n_estimators: int = Field(..., description="Number of estimators for the model")
    max_depth: int | None = Field(
        default=None, description="Maximum depth for the model"
    )
    random_state: int = Field(..., description="Random state for the model")
    deployable_threshold: float = Field(
        default=0.5, description="Threshold for deployment"
    )

    def _validate_data(self, data: pd.DataFrame, is_train: bool = True) -> None:
        """
        Validates the given DataFrame `data` for training or inference.

        Args:
            data (pd.DataFrame): The DataFrame to be validated.
            is_train (bool, optional): Flag indicating whether the data is for training. Defaults to True.

        Raises:
            HTTPException: If the DataFrame does not have column names or if the target column is not found for training.

        Returns:
            None
        """
        if data.columns.isnull().any():
            raise HTTPException(
                status_code=400, detail="Dataset must have column names"
            )

        if is_train and self.target not in data.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{self.target}' not found. Target column required for training data.",
            )

    def _save_model_locally(self, model: SklearnRandomForestRegressor) -> str:
        """
        Save the given Random Forest Regressor model locally.

        Parameters:
            model (SklearnRandomForestRegressor): The model to be saved.

        Returns:
            str: The file path where the model is saved.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
            pickle.dump(model, tmp_file)
            return tmp_file.name

    def train(self, train_data: pd.DataFrame) -> TrainModelResponse:
        """
        Trains a random forest regressor model on the given `train_data` and saves the model to a pickle file.

        Args:
            train_data (pd.DataFrame): The training data to be used for training the model.

        Returns:
            TrainModel: An instance of the `TrainModel` class containing the save path, model ID, and score of the trained model.

        Raises:
            HTTPException: If the `train_data` does not have column names or if the target column is not found for training.
        """
        model = SklearnRandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self._validate_data(train_data)
        X: pd.DataFrame = train_data.drop(columns=[self.target])
        y: pd.Series = train_data[self.target]

        model.fit(X, y)

        model_id = str(uuid4())

        save_path = self._save_model_locally(model)
        return TrainModelResponse(save_path=save_path, model_id=model_id)

    def _get_performance_metrics(
        self, y_true: ndarray, y_pred: ndarray
    ) -> PerformanceMetrics:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return PerformanceMetrics(
            mse=float(mse),
            rmse=float(rmse),
            r2=float(r2),
        )

    def test(
        self, model: SklearnRandomForestRegressor, test_data: pd.DataFrame
    ) -> TestModelResponse:
        self._validate_data(test_data, False)

        X_test: pd.DataFrame = test_data.drop(columns=[self.target])
        y_test: ndarray = test_data[self.target].to_numpy()

        try:
            y_pred: ndarray = model.predict(X_test)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to predict on test data. {e}"
            )

        metrics: PerformanceMetrics = self._get_performance_metrics(y_test, y_pred)
        is_deployable: bool = metrics.r2 < self.deployable_threshold

        return TestModelResponse(
            **metrics.model_dump(),
            deployable=is_deployable,
        )
