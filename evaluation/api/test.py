import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "board": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "time_control": '600+0',
    "white_remaining_time": 180,
    "black_remaining_time": 180,
    "white_rating": 1500,
    "black_rating": 1500
}

response = requests.post(url, json=data)
print(response.json())
