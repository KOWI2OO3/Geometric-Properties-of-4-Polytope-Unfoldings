
FROM ubuntu:22.04

# install app dependencies
# Install python en pip
RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install numpy
RUN pip install scipy

COPY ./src/*.py /
COPY requirements.txt .

RUN pip install -r requirements.txt

# ensure output-folder exists
RUN mkdir /output

CMD ["python3", "/pipeline.py"]