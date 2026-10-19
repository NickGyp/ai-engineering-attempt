
# AI Engineering Learning Project

A hands-on Python project exploring AI application development, model serving, and deep learning fundamentals.

## Sentiment Analysis API

A REST API built with FastAPI that uses a pretrained DistilBERT model from Hugging Face to classify English text sentiment as positive or negative.

### Features

- Sentiment analysis through a POST endpoint (`/sentiment`)
- Pretrained Transformer inference using PyTorch
- JSON request validation with Pydantic
- Automated API testing with pytest
- Interactive API documentation through FastAPI Swagger UI

## Technologies

Python, FastAPI, PyTorch, Hugging Face Transformers, Pydantic, pytest.

## Project Structure

- `main.py` — FastAPI application
- `sentiment_model.py` — Sentiment inference logic
- `test_api.py` — API test
- `test_model.py` — Initial model experiments
- `exercises/` — PyTorch learning exercises
- `requirements.txt` — Python dependencies

## Getting Started

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the sentiment endpoint.

## Running Tests

```bash
python -m pytest test_api.py -v
```

## Project Status

Educational project under active development. The sentiment analysis API is functional and has a passing automated test.
