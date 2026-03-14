FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
