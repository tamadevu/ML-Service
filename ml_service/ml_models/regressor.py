import pickle
from uuid import uuid4
from fastapi import HTTPException
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from ml_service.schemas.regressor import SaveModel, TrainModel


class Regressor(BaseModel):
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

    def _save_model(self, model: RandomForestRegressor) -> SaveModel:
        """
        Saves the given RandomForestRegressor model to a pickle file in the specified directory.

        Args:
            model (RandomForestRegressor): The model to be saved.

        Returns:
            str: The path of the saved pickle file.
        """
        model_id: str = str(uuid4())
        save_path: str = f"/home/luna/tamadevu/ml_service_models/{model_id}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

        return SaveModel(save_path=save_path, model_id=model_id)

    def train(self, train_data: pd.DataFrame, test_split: float = 0.2) -> TrainModel:
        """
        Trains a random forest regressor model on the given `train_data` and saves the model to a pickle file.

        Args:
            train_data (pd.DataFrame): The training data to be used for training the model.
            test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

        Returns:
            TrainModel: An instance of the `TrainModel` class containing the save path, model ID, and score of the trained model.

        Raises:
            HTTPException: If the `train_data` does not have column names or if the target column is not found for training.
        """
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        self._validate_data(train_data)
        X: pd.DataFrame = train_data.drop(columns=[self.target])
        y: pd.Series = train_data[self.target]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_split, random_state=self.random_state
        )

        model.fit(X_train, y_train)
        save_results: SaveModel = self._save_model(model)

        score: float = float(model.score(X_val, y_val))

        return TrainModel(
            save_path=save_results.save_path,
            model_id=save_results.model_id,
            score=score,
        )
