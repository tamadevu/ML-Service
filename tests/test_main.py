import pytest
from fastapi.testclient import TestClient
from ml_service.app.main import app
from ml_service.app.utils.schema_encoder_decoder import encode_data_schema

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}


def test_dataset():
    data_schema: dict = {
        "name": "str",
        "age": "int",
        "income": "float",
        "is_employed": "bool",
    }
    count: int = 5

    encoded_schema = encode_data_schema(data_schema)
    response = client.get(f"/dataset/{encoded_schema}/{count}")
    assert response.status_code == 200

    generated_data: list[dict] = response.json()

    assert len(generated_data) == count

    for datapoint in generated_data:
        assert "name" in datapoint
        assert "age" in datapoint
        assert "income" in datapoint
        assert "is_employed" in datapoint

        assert isinstance(datapoint["name"], str)
        assert isinstance(datapoint["age"], int)
        assert isinstance(datapoint["income"], float)
        assert isinstance(datapoint["is_employed"], bool)


def test_dataset_invalid_type():
    data_schema: dict = {"name": "unsupported_type"}

    count: int = 1
    encoded_schema = encode_data_schema(data_schema)
    response = client.get(f"/dataset/{encoded_schema}/{count}")
    assert response.status_code == 400
    assert response.json() == {
        "detail": "Data type unsupported_type not supported. Please use one of the following: str, int, bool, float"
    }


if __name__ == "__main__":
    pytest.main()
