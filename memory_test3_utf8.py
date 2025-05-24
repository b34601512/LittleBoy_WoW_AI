import requests
url = 'http://localhost:8765/api/v1/memories/?user_id=deshraj'
headers = {'Authorization': 'Bearer sk-cb13565e1c894d54af4f47e04ddee800', 'Content-Type': 'application/json'}
print('测试读取记忆')
resp = requests.get(url, headers=headers)
print('状态码:', resp.status_code)
print('响应:', resp.text)
