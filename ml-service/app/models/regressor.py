import pickle
from fastapi import HTTPException
from numpy import ndarray
import pandas as pd
import numpy as np
from schemas.data import LabelledDataSchema, DataSchema
from schemas.model import PerformanceMetrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pydantic import ValidationError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



    
class Model:
    def __init__(self, target: str, n_estimators: int, max_depth:int|None,random_state:int )-> None:
        """
        Initializes the class with the given parameters.

        Args:
            target (str): The target variable.
            n_estimators (int): The number of estimators.
            max_depth (int|None): The maximum depth of the tree.
            random_state (int): The random state.

        Returns:
            None
        """
        self.target = target
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _validate_train_data(self, train_data: pd.DataFrame):
        """
        Validates the training data by checking if each row conforms to the LabelledDataSchema.
        
        Args:
            train_data (pd.DataFrame): The DataFrame containing the training data.
        
        Raises:
            HTTPException: If any row in the training data fails validation.
        
        Returns:
            None
        """
        errors: list[dict] = []
        
        for index, data in train_data.iterrows():
            try: 
                LabelledDataSchema(**data.to_dict())
            except ValidationError as e:
                errors.append({"row": index, "errors": e.errors()})
    
        if errors:
            raise HTTPException(status_code=400, detail=errors)
        
    def _validate_test_data(self, test_data: pd.DataFrame):
        """
        Validates the test data by checking if each row conforms to the DataSchema.

        Args:
            test_data (pd.DataFrame): The DataFrame containing the test data.

        Raises:
            HTTPException: If any row in the test data fails validation.

        Returns:
            None

        This function iterates over each row in the test data and tries to create a DataSchema object using the data.
        If any row fails validation, the row index and the validation errors are added to the `errors` list.
        If the `errors` list is not empty, an HTTPException is raised with a status code of 400 and the `errors` list as the detail.
        """
        errors: list[dict] = []
        
        for index, data in test_data.iterrows():
            try: 
                DataSchema(**data.to_dict())
            except ValidationError as e:
                errors.append({"row": index, "errors": e.errors()})
    
        if errors:
            raise HTTPException(status_code=400, detail=errors)
    

    def _train_val_split(self, train_data: pd.DataFrame, test_split: int | float = 0.2) -> list:
        """
        Splits the given training data into training and validation sets.

        Parameters:
            train_data (pd.DataFrame): The training data to be split.
            target (str, optional): The name of the target column. Defaults to "income".
            test_split (int | float, optional): The proportion of the data to include in the validation set. Defaults to 0.2.

        Returns:
            list: A list containing the training and validation sets. The order of the elements in the list is as follows:
                - X_train (pd.DataFrame): The training features.
                - X_val (pd.Series): The validation features.
                - y_train (pd.DataFrame): The training labels.
                - y_val (pd.Series): The validation labels.
        """
        
        X: pd.DataFrame = train_data.drop(columns=[self.target])
        y: pd.Series = train_data[self.target]
       
        return train_test_split(X, y, test_size=test_split, random_state=self.random_state)
    
    def _save_model(self, model: RandomForestRegressor):
        """
        Saves a trained RandomForestRegressor model to a file.

        Parameters:
            model (RandomForestRegressor): The trained RandomForestRegressor model to be saved.

        Returns:
            None
        """
        model_path = f'model_{self.random_state}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
    def train(self, train_data: pd.DataFrame, test_split: int | float = 0.2) -> str:
        """
        Trains a random forest regressor model on the given training data.

        Parameters:
            train_data (pd.DataFrame): The training data to fit the model on.
            target (str, optional): The name of the target variable. Defaults to "income".
            test_split (int | float, optional): The proportion of the data to use for validation. Defaults to 0.2.

        Returns:
            str: A message containing the model ID and the accuracy score of the model on the evaluation data.
        """
        model: RandomForestRegressor = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

        self._validate_train_data(train_data)
        X_train, X_val, y_train, y_val = self._train_val_split(train_data, test_split)

        model.fit(X_train, y_train)
        self._save_model(model)

        score: float = float(model.score(X_val, y_val))
        model_id: str =""
        
        return f"Your model id is {model_id}\n Your model's accuracy score on eval data is: {score}"


    def predict(self, model: RandomForestRegressor, data: pd.DataFrame)-> pd.DataFrame:
        """
        Predicts the target variable values for a given data set using a trained RandomForestRegressor model.

        Parameters:
            model (RandomForestRegressor): The trained RandomForestRegressor model.
            data (pd.DataFrame): The data set for which the target variable values are to be predicted.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted target variable values along with their lower and upper bounds.
                The DataFrame has the following columns:
                - "Prediction": The predicted target variable values.
                - "Lower Bound": The lower bound of the confidence interval for the predicted target variable values.
                - "Upper Bound": The upper bound of the confidence interval for the predicted target variable values.
        """
       
        self._validate_test_data(data)
        predictions = np.array([tree.predict(data) for tree in model.estimators_])


        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        confidence_interval = 1.96 * std_prediction

        results_df = pd.DataFrame({
            "Prediction": mean_prediction,
            "Lower Bound": mean_prediction - confidence_interval,
            "Upper Bound": mean_prediction + confidence_interval
        })

        return results_df


    
    def test(self, model: RandomForestRegressor, test_data: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate the performance metrics for a given test dataset.

        Args:
            test_data (pd.DataFrame): The test dataset to evaluate the model on.

        Returns:
            PerformanceMetrics: An object containing the calculated performance metrics:
                - MAE (float): Mean Absolute Error
                - MSE (float): Mean Squared Error
                - RMSE (float): Root Mean Squared Error
                - R2 (float): R-squared score
        """
        self._validate_test_data(test_data)
        X_test: pd.DataFrame = test_data.drop(columns=[self.target])
        y_test: pd.Series = test_data[self.target]

        y_pred: ndarray = model.predict(X_test)
       
        
        mae = mean_absolute_error(y_test, y_pred, multioutput="uniform_average")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        is_deployable= bool(mae < 10 and rmse < 15 and r2 > 0.7)

        return PerformanceMetrics(mae=float(mae), mse=float(mse), rmse=float(rmse), r2=float(r2), deployable=is_deployable)
