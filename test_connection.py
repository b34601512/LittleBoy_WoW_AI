import json

# 测试OpenMemory API连接
url = "http://localhost:8765/api/healthcheck"  # 健康检查端点
try:
    resp = requests.get(url)
    print(f"服务器状态码: {resp.status_code}")
    print(f"服务器响应: {resp.text}")
    
    if resp.status_code == 200:
        print("✓ OpenMemory服务器连接正常")
    else:
        print("⚠ OpenMemory服务器响应异常")
except Exception as e:
    print(f"连接错误: {str(e)}")