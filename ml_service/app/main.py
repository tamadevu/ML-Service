import base64
import json
from fastapi import FastAPI, HTTPException
from uuid import uuid4
import random
from ml_service.schemas.dataset import CreateDatasetSchema, DataSchema
@app.post("/dataset")
async def dataset(body: CreateDatasetSchema):
    """
    Endpoint to generate random data based on the provided data schema.

    Args:
        data_schema (str): Base64 encoded JSON string representing the data schema.
        count (int): Number of data points to generate.

    Returns:
        list[dict]: List of generated data points.

    Raises:
        HTTPException: If the data schema is invalid or contains unsupported data types.
    """

    data_schema: list[DataSchema] = body.data_schema
    count: int = body.count

    generated_data: list[dict] = []

    for _ in range(count):
        random_datapoint: dict = {}
        for item in data_schema:
            column = item.name
            match item.type:
                case "str":
                    random_datapoint[column] = str(uuid4())
                case "int":
                    random_datapoint[column] = random.randint(0, 100)
                case "float":
                    random_datapoint[column] = round(random.random(), 2)
                case "bool":
                    random_datapoint[column] = random.choice([False, True])
                case _:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Data type {item.type} not supported. Please use one of the following: str, int, bool, float",
                    )
        generated_data.append(random_datapoint)

    return generated_data
