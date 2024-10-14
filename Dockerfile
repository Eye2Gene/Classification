# Use an official TensorFlow image as a parent image
FROM tensorflow/tensorflow:2.10.0-gpu

RUN echo "alias ltr='ls -ltr'" >> /etc/bash.bashrc
RUN pip install --no-cache-dir keras_cv_attention_models

WORKDIR /app
COPY . /app

RUN mkdir -p trained_models
RUN pip install --no-cache-dir -r requirements.txt
