import base64
import json


def encode_data_schema(data_schema: dict) -> str:
    schema_str = json.dumps(data_schema)
    encoded_schema = base64.urlsafe_b64encode(schema_str.encode("utf-8")).decode(
        "utf-8"
    )
    return encoded_schema


def decode_data_schema(data_schema: dict) -> str:
    decoded_schema = base64.urlsafe_b64decode(data_schema).decode("utf-8")
    data_schema_dict = json.loads(decoded_schema)
    return data_schema_dict
