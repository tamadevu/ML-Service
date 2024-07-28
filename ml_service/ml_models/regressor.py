import pickle
from uuid import uuid4
from fastapi import HTTPException
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from pydantic import BaseModel, Field
import tempfile

from ml_service.schemas.regressor import TrainModelResponse


class RandomForestRegressor(BaseModel):
    target: str = Field(..., description="Target column name")
    n_estimators: int = Field(..., description="Number of estimators for the model")
    max_depth: int | None = Field(
        default=None, description="Maximum depth for the model"
    )
    random_state: int = Field(..., description="Random state for the model")

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
                status_code=400, detail=f"Target column '{self.target}' not found"
            )

    def _save_model_locally(self, model: SklearnRandomForestRegressor) -> str:
        """
        Save the given Random Forest Regressor model locally.

        Parameters:
            model (SklearnRandomForestRegressor): The model to be saved.

        Returns:
            str: The file path where the model is saved.
        """
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            pickle.dump(model, tmp_file)
            tmp_file_path = tmp_file.name
        return tmp_file_path

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
