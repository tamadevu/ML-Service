import pytest
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestRegressor
from ml_service.ml_models.regressor import Regressor
from ml_service.schemas.regressor import SaveModel, TrainModel
from unittest.mock import mock_open, patch


@pytest.fixture
def regressor():
    return Regressor(target="target", n_estimators=10, max_depth=5, random_state=42)


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


@patch("builtins.open", new_callable=mock_open)
@patch("pickle.dump")
def test_save_model(mock_pickle_dump, mock_open_file, regressor):
    model = RandomForestRegressor()
    result = regressor._save_model(model)
    assert isinstance(result, SaveModel)
    assert mock_open_file.called
    assert mock_pickle_dump.called


@patch.object(
    Regressor,
    "_save_model",
    return_value=SaveModel(save_path="path/to/model.pkl", model_id="12345"),
)
@patch("ml_service.ml_models.regressor.RandomForestRegressor.fit")
@patch("ml_service.ml_models.regressor.RandomForestRegressor.score")
def test_train_model(mock_save_model, mock_score, mock_fit, regressor, valid_data):
    result = regressor.train(valid_data)
    assert isinstance(result, TrainModel)
    assert mock_fit.called
    assert mock_save_model.called
    assert mock_score.called


if __name__ == "__main__":
    pytest.main()
