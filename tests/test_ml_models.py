import pytest
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor
from ml_service.ml_models.regressor import RandomForestModel
from ml_service.schemas.regressor import TrainModelResponse
from unittest.mock import patch


@pytest.fixture
def regressor():
    return RandomForestModel(
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
    model = RandomForestRegressor()
    result = regressor._save_model_locally(model)
    assert isinstance(result, str)
    assert mock_pickle_dump.called


@patch.object(
    RandomForestModel,
    "_save_model_locally",
    return_value="path/to/model.pkl",
)
@patch("ml_service.ml_models.regressor.RandomForestRegressor.fit")
def test_train_model(mock_save_model, mock_fit, regressor, valid_data):
    result = regressor.train(valid_data)
    assert isinstance(result, TrainModelResponse)
    assert mock_fit.called
    assert mock_save_model.called


if __name__ == "__main__":
    pytest.main()
