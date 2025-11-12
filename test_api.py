import json
from app import app

client = app.test_client()

def test(id_val=4):
    resp = client.post('/api/predict', json={'patient_id': id_val, 'id_col': 'id'})
    print('STATUS:', resp.status_code)
    try:
        data = resp.get_json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.get_data(as_text=True))

if __name__ == '__main__':
    test(4)
