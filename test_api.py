from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_sentiment():
    response = client.post(
        "/sentiment",
        json = {"text": "I love python!"}
    )

    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIVE"

def test_empty_text():
    response = client.post(
        "/sentiment",
        json={"text": ""}
    )

    assert response.status_code == 422

def test_whitespace_text():
    response = client.post(
        "/sentiment",
        json={"text": "   "}
    )

    assert response.status_code == 422