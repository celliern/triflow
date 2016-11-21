FROM python:3

RUN apt-get update
RUN apt-get install -y build-essential gfortran

# RUN pip install dask
# RUN pip install numpy
# RUN pip install appdirs
# RUN pip install bokeh
# RUN pip install datreant.core
# RUN pip install sympy
# RUN pip install scipy
# RUN pip install path.py
# RUN pip install click

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /usr/src/app

RUN python /usr/src/app/setup.py install
RUN triflow_cache_simple

CMD [ "dask-worker", "scheduler:8786"]
