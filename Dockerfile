FROM python:3.12.5-slim

WORKDIR /app

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install libgomp1

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt