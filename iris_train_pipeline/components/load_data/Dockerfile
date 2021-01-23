FROM python:3.7-slim

WORKDIR /app

RUN pip install requirements.txt

COPY src ./

ENTRYPOINT [ "python", "src/load_data.py" ]