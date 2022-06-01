import requests


url = f'http://160.85.252.53:5000/api/question/world_cup_data_v2'

headers = {
    'content-type': 'application/json',
    "x-api-key": "60106d9e76d0-43259aa49ea9b4cc83b7",
}
res = requests.put(
    url,
    headers=headers,
    json={
        'query': "select capacity, stadium_name from stadium group by stadium_name, capacity order by max(capacity) desc limit 1;",
        'beam_size': 2,
        "db_id": "world_cup_data_v2"
    }
)

print(res.json())