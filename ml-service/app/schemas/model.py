from pydantic import BaseModel, Field

class PerformanceMetrics(BaseModel):
    mae: float = Field(description="Mean Absolute Error")
    mse: float = Field(description="Mean Squared Error")
    rmse: float = Field(description="Root Mean Squared Error")
    r2: float = Field(description="R-squared")
    deployable: bool = Field(description="Is model deployable?")
