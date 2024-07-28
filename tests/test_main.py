import pytest
from fastapi.testclient import TestClient
from ml_service.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_dataset():
    """
    Test the dataset endpoint by generating random data based on a given data schema.

    This test case verifies that the dataset endpoint returns the expected number
    of data points and that each data point has the correct structure and data types.
    """
    # Define the data schema
    data_schema = [{"name": "Name", "type": "str"}, {"name": "Age", "type": "int"}]
    count: int = 5
    payload: dict = {"data_schema": data_schema, "count": count}

    response = client.post("/dataset", json=payload)

    assert response.status_code == 200

    generated_data: list[dict] = response.json()
    assert len(generated_data) == count

    for datapoint in generated_data:
        assert "Name" in datapoint
        assert "Age" in datapoint

        assert isinstance(datapoint["Name"], str)
        assert isinstance(datapoint["Age"], int)


def test_dataset_invalid_type():
    """
    Test the dataset endpoint by requesting random data with an unsupported data type.

    This test case verifies that the dataset endpoint returns a 400 status code and
    a relevant error message when an unsupported data type is specified in the data schema.

    Args:
        None

    Returns:
        None
    """
    data_schema = [{"name": "Name", "type": "unsupported_type"}]
    count: int = 1

    payload: dict = {"data_schema": data_schema, "count": count}
    response = client.post("/dataset", json=payload)

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Data type unsupported_type not supported. Please use one of the following: str, int, bool, float"
    }


if __name__ == "__main__":
    pytest.main()
