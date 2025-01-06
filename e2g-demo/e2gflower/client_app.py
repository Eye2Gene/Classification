"""e2gflower: A Flower / E2G app."""

import logging
import os

from e2gflower.task import load_latest_model_from_s3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import subprocess

import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Suppress other warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["TOWER_ACCESS_TOKEN"] = "eyJ0aWQiOiAxMDI0OX0uZWNkZDlhNTQwMmZmODhhYjFmNGY4OGU0MjU4NTI2Nzc2ZjMyYzNlYg=="
print(f">>> debug: ENV {os.environ}")


# Define Flower client
class FlowerClient(NumPyClient):
    """Flower client for federated learning with S3 and Nextflow integration."""

    def __init__(self, s3_bucket, s3_key_prefix, nextflow_command):
        """Initialize the FlowerClient with S3 bucket, key prefix, and Nextflow command.

        Args:
            s3_bucket (str): The S3 bucket name.
            s3_key_prefix (str): The prefix for the S3 keys.
            nextflow_command (str): The command to run the Nextflow pipeline.
        """
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.nextflow_command = nextflow_command

    def fit(self, parameters, config):  # must use 3 arguments
        """Run the Nextflow pipeline and get the latest model from S3.

        Args:
            parameters (list): The model parameters.
            config (dict): Configuration for the training process.

        Returns:
            tuple: A tuple containing model parameters, number of examples, and metrics.
        """
        _, _, initial_model_path_S3 = load_latest_model_from_s3("eye2gene-main", "initial_model")
        nextflow_command = self.nextflow_command + f" --load_weights_h5_path '{initial_model_path_S3}'"
        print(f">>> debug: 'def fit' nextflow_command {nextflow_command}")
        # Run Nextflow pipeline
        subprocess.run(nextflow_command, shell=True, check=True)

        # Get the latest model from S3
        model, metadata, _ = load_latest_model_from_s3(self.s3_bucket, self.s3_key_prefix)

        # Return model parameters, number of examples, and metrics
        return model.get_weights(), 16709, metadata

    def evaluate(self, parameters, config):  # must use 3 arguments
        """Evaluate the model using the latest metrics from S3."""
        # For this setup, evaluation is done within the Nextflow pipeline
        # We'll return the metrics from the latest model
        _, metadata = load_latest_model_from_s3(self.s3_bucket, self.s3_key_prefix)
        print(f">>> debug: 'def evaluate' metadata {metadata}")
        return metadata["loss"], 16709, {"accuracy": metadata["accuracy"]}


def client_fn(context: Context):
    """Create a Flower client based on the given context.

    Args:
        context (Context): The context containing node configuration.

    Returns:
        FlowerClient: An instance of FlowerClient.
    """
    print(f">>> debug: 'def client_fn' context {context}")
    try:
        # Get site from context
        site = context.node_config["site"]

        # Prepare configurations
        s3_bucket = f"eye2gene-main-{site}"
        s3_key_prefix = "out/1/trained_models/"
        nextflow_profile = f"eye2gene_{site}"

        # Construct Nextflow command
        nextflow_command = (
            f"nextflow run Eye2Gene/Classification -r main "
            f"-c ~/.nextflow/assets/Eye2Gene/Classification/aws_params.config "
            f"-profile {nextflow_profile} -with-tower"
        )

        return FlowerClient(s3_bucket, s3_key_prefix, nextflow_command).to_client()

    except KeyError as e:
        raise ValueError(f"Missing required context key: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred while creating the Flower client: {e}")


app = ClientApp(client_fn=client_fn)
