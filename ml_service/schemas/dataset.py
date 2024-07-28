from pydantic import BaseModel


class DataSchema(BaseModel):
    type: str
    name: str


class CreateDatasetSchema(BaseModel):
    data_schema: list[DataSchema]
    count: int
