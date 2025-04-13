# 使用 Python 3.9 的精簡映像
FROM python:3.9-slim

# 避免互動安裝與 locale 問題
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 安裝系統套件（避免 gradio_client 等套件報錯）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 設定工作資料夾
WORKDIR /app

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY . .

# 避免啟動失敗時 container 馬上退出（如 StaticFiles 404）
EXPOSE 8000

# 啟動 FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]