import requests

url = "http://localhost:8765/api/v1/memories/?user_id=deshraj"
headers = {
    "Authorization": "Bearer sk-cb13565e1c894d54af4f47e04ddee800",
    "Content-Type": "application/json"
}
resp = requests.get(url, headers=headers)
print(resp.status_code)
print(resp.text)