from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: str
    is_tumor: bool
    status: str = "success"