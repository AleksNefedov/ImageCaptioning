FROM python:3.9
RUN pip install --upgrade pip

WORKDIR /project

COPY ./checkpoints ./checkpoints
COPY ./static ./static
COPY ./templates ./templates
COPY ./images ./images
COPY ./data ./data
COPY ./libs ./libs
COPY ./mlflow ./mlflow
COPY ./misc ./misc

COPY server.py .
COPY requirements.txt .
COPY start_flask_app.sh .
COPY start_celery_worker.sh .
COPY start_jupyter_lab.sh .
COPY start_mlflow_server.sh .
COPY README.md .

RUN pip install -r requirements.txt
