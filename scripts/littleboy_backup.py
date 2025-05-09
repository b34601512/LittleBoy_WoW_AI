import os

# 设置目标目录和输出文件路径
root_dir = r'D:\wow_ai'
output_file = r'D:\wow_ai\all_code_backup.txt'

with open(output_file, 'w', encoding='utf-8') as outfile:
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(foldername, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(f"\n\n# ===== 文件: {file_path} =====\n")
                        outfile.write(infile.read())
                except Exception as e:
                    print(f"无法读取文件 {file_path}：{e}")

print(f"所有Python代码已备份到：{output_file}")
