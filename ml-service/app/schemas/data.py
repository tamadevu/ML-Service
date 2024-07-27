from pydantic import BaseModel

class AgeSchema(BaseModel):
    type: int
    range: dict[str, int]

class DataSchema(BaseModel):
    name: str
    age: AgeSchema

class LabelledDataSchema(DataSchema):
    income: float

class DataGenerationSchema(BaseModel):
    schema: LabelledDataSchema
    number: int



