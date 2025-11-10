FROM python:3.9-slim-bullseye

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ALL files first
COPY . /app

# Then install requirements (now setup.py is available)
RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]