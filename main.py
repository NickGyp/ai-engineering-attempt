from fastapi import FastAPI
from pydantic import BaseModel, constr
from sentiment_model import predict_sentiment

app = FastAPI()

class TextRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1)

@app.post("/sentiment")
def sentiment(request: TextRequest):
    prediction = predict_sentiment(request.text)

    return {
        "text": request.text,
        **prediction
    }