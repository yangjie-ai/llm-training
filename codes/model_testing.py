import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "人工智能是什么",
        "max_length": 1024,
        "temperature": 0.7
    }
)
print(response.json()["generated_text"])