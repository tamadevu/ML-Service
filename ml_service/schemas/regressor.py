from pydantic import BaseModel


class SaveModel(BaseModel):
    save_path: str
    model_id: str


class TrainModel(SaveModel):
    score: float
