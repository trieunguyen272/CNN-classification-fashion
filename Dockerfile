# FROM python:3.9
FROM jjanzic/docker-python3-opencv
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN apt-get update -y 
RUN apt update; apt install -y libgl1

ENTRYPOINT [ "python" ]

CMD ["app.py"]