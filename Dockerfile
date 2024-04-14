FROM python:3.10

WORKDIR /Titanic_ML

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN export PYTHONPATH='${PYTHONPATH}:/Titanic_ML'

COPY . .

CMD ["python", "./app.py"]