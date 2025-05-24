import requests
print('测试连接服务器')
resp = requests.get('http://localhost:8765/api/healthcheck')
print('状态码:', resp.status_code)
print('响应:', resp.text)
