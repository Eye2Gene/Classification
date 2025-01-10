# Use an official TensorFlow image as a parent image
FROM tensorflow/tensorflow:2.15.0-gpu

# Set up system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set up bash alias
RUN echo "alias ltr='ls -ltr'" >> /etc/bash.bashrc

# Set working directory
WORKDIR /app

# Copy the rest of the application
COPY . .

# Copy only pyproject.toml and poetry.lock (if it exists)
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Create directory for trained models
RUN mkdir -p trained_models
