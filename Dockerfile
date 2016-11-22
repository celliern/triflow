FROM python:3

RUN apt-get update
RUN apt-get install -y build-essential gfortran

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app

RUN python /usr/src/app/setup.py install
RUN triflow_cache_simple && triflow_cache_full

CMD [ "dask-worker", "scheduler:8786"]
