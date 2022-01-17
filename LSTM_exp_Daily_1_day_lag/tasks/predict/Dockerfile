FROM model_pipeline_base:latest

RUN apt-get update
RUN apt-get install -y libsnappy-dev

COPY predict.py /

WORKDIR /
