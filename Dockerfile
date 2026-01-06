FROM python:3.9-slim-bullseye 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/

COPY results/ /app/model/

ENTRYPOINT ["python", "-m", "src.predict"]
