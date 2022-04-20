import requests, json
from requests.auth import HTTPBasicAuth

headers = {
    'content-type': 'application/json',
    "x-api-key": "1234",
}

auth = HTTPBasicAuth('apikey', '1234')
url = 'http://localhost:5000/api/question/world_cup_data_v2'
res = requests.put(
    url,
    headers=headers,
    json={
        'question': 'Return the player that played the most games?',
        'beam_size': 1
    }
)


print(res.json())
