# Use an official TensorFlow image as a parent image
FROM tensorflow/tensorflow:2.15.0.post1-gpu

RUN echo "alias ltr='ls -ltr'" >> /etc/bash.bashrc

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir keras_cv_attention_models
RUN mkdir -p trained_models
RUN mkdir -p checkpoints
RUN mkdir -p logs
