@echo off
chcp 65001 >nul

:: 设置 Git 安全目录（可多次执行，防止 Git 提示安全问题）
git config --global --add safe.directory "D:/wow_ai" 2>nul

:: 进入 wow_ai 项目目录
cd /d "D:\wow_ai"

:: 添加所有 .py 文件（会递归处理子目录，且遵循 .gitignore）
git add "*.py"

:: 获取当前日期作为提交信息
for /f %%i in ('powershell -command "Get-Date -Format yyyy-MM-dd"') do set DATE=%%i
git commit -m "备份 %DATE% 的 Python 源码"

:: 推送到远程仓库 main 分支
git push origin main

:: 完成提示
echo.
echo ✅ 所有 .py 文件备份完成！按任意键关闭窗口...
pause >nul
