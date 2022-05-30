import requests

headers = {
    'content-type': 'application/json',
}

url = f'http://localhost:5000/api/question/cordis'

res = requests.put(
    url,
    headers=headers,
    json={
        'query': "select pmr.code from projects as p join project_members as pm on p.unics_id = pm.project join project_member_roles as pmr on pm.member_role = pmr.code where p.acronym = 'ARGOS' and pm.member_name like '%ATHENA%'",
        'beam_size': 2,
        "db_id": "cordis_temporary"
    }
)

print(res.json())