import pytest
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from ml_service.ml_models.regressor import RandomForestRegressor
from ml_service.schemas.regressor import (
    PerformanceMetrics,
    TrainModelResponse,
    TestModelResponse,
)
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def regressor():
    return RandomForestRegressor(
        target="target", n_estimators=10, max_depth=5, random_state=42
    )


@pytest.fixture
def valid_data():
    return pd.DataFrame(
        {"feature_1": [1, 2, 3], "feature_2": [4, 5, 6], "target": [7, 8, 9]},
        index=None,
    )


@pytest.fixture
def invalid_data_no_columns():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[None, None, None])


def test_validate_data_no_columns(regressor, invalid_data_no_columns):
    with pytest.raises(HTTPException) as excinfo:
        regressor._validate_data(invalid_data_no_columns)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Dataset must have column names"


def test_validate_data_valid(regressor, valid_data):
    try:
        regressor._validate_data(valid_data)
    except HTTPException:
        pytest.fail("Unexpected HTTPException raised")


@patch("pickle.dump")
def test_save_model(mock_pickle_dump, regressor):
    model = SklearnRandomForestRegressor()
    result = regressor._save_model_locally(model)
    assert isinstance(result, str)
    assert mock_pickle_dump.called


@patch.object(
    RandomForestRegressor,
    "_save_model_locally",
    return_value="path/to/model.pkl",
)
@patch("ml_service.ml_models.regressor.SklearnRandomForestRegressor.fit")
def test_train_model(mock_save_model, mock_fit, regressor, valid_data):
    result = regressor.train(valid_data)
    assert isinstance(result, TrainModelResponse)
    assert mock_fit.called
    assert mock_save_model.called


def test_get_performance_metrics(regressor):
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])
    expected_metrics = PerformanceMetrics(mse=1.0, rmse=1.0, r2=-0.5)
    assert regressor._get_performance_metrics(y_true, y_pred) == expected_metrics


@patch.object(
    RandomForestRegressor,
    "_get_performance_metrics",
    return_value=PerformanceMetrics(mse=1.0, rmse=1.0, r2=-0.5),
)
@patch(
    "ml_service.ml_models.regressor.SklearnRandomForestRegressor.predict",
    return_value=np.array([1.0, 2.0, 3.0]),
)
def test_test_model(mock_save_model, mock_predict, regressor, valid_data):
    mock_model = MagicMock(spec=SklearnRandomForestRegressor)

    result = regressor.test(mock_model, valid_data)
    assert isinstance(result, TestModelResponse)
    assert mock_predict.called


if __name__ == "__main__":
    pytest.main()
