from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_model import predict_sentiment

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment(request: TextRequest):
    prediction = predict_sentiment(request.text)

    return {
        "text": request.text,
        **prediction
    }