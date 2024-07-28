from pydantic import BaseModel, Field


class TrainModelResponse(BaseModel):
    save_path: str
    model_id: str


class PerformanceMetrics(BaseModel):
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    r2: float = Field(description="R-squared")


class TestModelResponse(PerformanceMetrics):
    deployable: bool = Field(description="Is model deployable?")
