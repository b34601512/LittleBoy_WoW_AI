import requests

# 设置 OpenMemory API 端点 URL
openmemory_url = "http://localhost:8765/api/v1/memories/"

# 设置 API Key
api_key = "sk-cb13565e1c894d54af4f47e04ddee800"  # 修改为你的实际 API Key

# 读取完整的 Markdown 内容（路径使用英文）
file_path = r"D:\chatgpt_backup\structured_content_here\test.txt"
with open(file_path, "r", encoding="utf-8") as file:
    md_content = file.read()

# 构建内存条目的数据
memory_data = {
    "user_id": "deshraj",  # 可以根据需要修改
    "text": md_content,  # 将Markdown内容作为内存条目文本
    "tags": ["WoW_Project", "AI_Automation", "Game_Agent", "OpenMemory"],  # 添加相关标签
    "metadata": {
        "source": "README_full_content.md",  # 文档来源
        "length": len(md_content.splitlines()),  # 文档行数作为长度
        "topic": "WoW Project AI Automation"
    }
}

# 设置请求头，带上你的 API 密钥
headers = {
    "Authorization": f"Bearer {api_key}",  # 修改为你的实际 API Key
    "Content-Type": "application/json"
}

# 发送 POST 请求到 OpenMemory
response = requests.post(openmemory_url, json=memory_data, headers=headers)

# 检查响应
if response.status_code == 200:
    print("Memory created successfully!")  # 成功时打印成功消息
else:
    print(f"Error: {response.status_code} - {response.text}")  # 错误时打印错误信息
