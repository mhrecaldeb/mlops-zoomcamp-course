FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipefile", "Pipefile.lock", "./" ]

RUN pipenv install --system --deploy

COPY ["predict_docker_hw.py", "./"]

CMD [ "python", "predict_docker_hw.py"]
