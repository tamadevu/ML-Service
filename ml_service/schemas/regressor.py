from pydantic import BaseModel


class TrainModelResponse(BaseModel):
    save_path: str
    model_id: str
