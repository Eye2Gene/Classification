FROM flwr/supernode:nightly

ENV TF_CPP_MIN_LOG_LEVEL=3

RUN pip install --no-cache \
    "flwr==1.14.0" \
    "boto3>=1.26.0" \
    "botocore>=1.29.0" \
    "tensorflow-cpu==2.15.1"

USER root
RUN apt-get update && apt-get install -y curl openjdk-11-jdk

USER app
RUN curl -s https://get.nextflow.io | bash

USER root
RUN mv nextflow /usr/local/bin/

USER app
RUN nextflow pull Eye2Gene/Classification
