@echo off
chcp 65001 >nul
title EduRAG 智能问答系统

echo ============================================
echo     EduRAG - 教育领域智能问答系统
echo     正在启动...
echo ============================================
echo.

:: 设置模型路径
set HF_HOME=D:\Projects\Models\huggingface
set HF_HUB_OFFLINE=1
set MODELSCOPE_CACHE=D:\Projects\Models\modelscope
set PYTHONUTF8=1

:: 检查 Redis 是否已运行
tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    echo [1/3] 启动 Redis...
    start "Redis" /MIN "D:\Projects\Redis\redis-server.exe" "D:\Projects\Redis\redis-edurag.conf"
    timeout /T 2 /NOBREAK >NUL
) else (
    echo [1/3] Redis 已运行，跳过
)

:: 切换到项目目录
cd /d D:\Projects\Educational_RAG_System

:: 启动 Web 服务
echo [2/3] 启动 Web 服务...
echo.
echo 正在加载模型（首次启动约需 30 秒）...
echo.
start "EduRAG" /MIN "C:\Users\pao03\AppData\Roaming\Python\Python314\Scripts\uv.exe" run --no-dev uvicorn app:app --host 127.0.0.1 --port 8001

:: 等待服务启动
echo [3/3] 等待服务就绪...
timeout /T 10 /NOBREAK >NUL

:: 打开浏览器
start http://127.0.0.1:8001

echo.
echo ============================================
echo     启动完成！
echo     访问 http://127.0.0.1:8001
echo     关闭本窗口即可停止服务
echo ============================================
echo.

:: 等待用户按键
pause
