import base64
import json
from fastapi import FastAPI, HTTPException
from uuid import uuid4
import random
from ml_service.app.utils.schema_encoder_decoder import decode_data_schema

@app.get("/dataset/{data_schema}/{count}")
async def dataset(data_schema: str, count: int):
    try:
        data_schema_dict = decode_data_schema(data_schema)
    except (json.JSONDecodeError, base64.binascii.Error):
        raise HTTPException(status_code=400, detail="Invalid JSON schema")

    generated_data: list[dict] = []

    for _ in range(count):
        random_datapoint: dict = {}
        for column, _type in data_schema_dict.items():
            match _type:
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
                        detail=f"Data type {_type} not supported. Please use one of the following: str, int, bool, float",
                    )
        generated_data.append(random_datapoint)

    return generated_data
