FROM flwr/superexec:1.11.0-py3.11-ubuntu22.04

RUN pip install --no-cache \
    "flwr==1.13.0" \
    # "flwr-datasets[vision]>=0.3.0" \
    # "monai==1.3.2" \
    "filelock==3.15.4"
