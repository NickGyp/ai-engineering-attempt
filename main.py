from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
class TextRequest(BaseModel):
    text: str
    language: str = "en"

classifier = pipeline(
    task = "sentiment-analysis",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

@app.get("/")
def home():
    return {"message":"Welcome to my first AI API effort"}

@app.get("/predict")
def predict(text: str):
    return {
        "input": text,
        "length": len(text)
    }

@app.get("/square")
def square(number: int):
    return {
        "number": number,
        "square": number**2
    }

@app.post("/analyze")
def analyze(request: TextRequest):
    words = request.text.split()

    return{
        "text": request.text,
        "language": request.language,
        "word_count": len(words)
    }

@app.post("/sentiment")
def sentiment(request: TextRequest):
    result = classifier(request.text)[0]

    return {
        "text": request.text,
        "sentiment": result["label"],
        "score": result["score"]
    }