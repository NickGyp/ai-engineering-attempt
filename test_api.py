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