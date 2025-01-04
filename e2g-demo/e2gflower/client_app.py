"""e2gflower: A Flower / E2G app."""

import csv
import json
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import subprocess

import boto3
import tensorflow as tf
from botocore.exceptions import ClientError
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

    def fit(self, parameters, config):
        """Run the Nextflow pipeline and get the latest model from S3.

        Args:
            parameters (list): The model parameters.
            config (dict): Configuration for the training process.

        Returns:
            tuple: A tuple containing model parameters, number of examples, and metrics.
        """
        # Run Nextflow pipeline
        subprocess.run(self.nextflow_command, shell=True, check=True)

        # Get the latest model from S3
        model, model_info, metrics = self.get_latest_model_from_s3()

        if model is None:
            raise ValueError("No model found in S3 after Nextflow run")

        # Return model parameters, number of examples, and metrics
        return self.get_parameters(model), 16709, metrics

    def evaluate(self, parameters, config):
        """Evaluate the model using the latest metrics from S3.

        Args:
            parameters (list): The model parameters.
            config (dict): Configuration for the evaluation process.

        Returns:
            tuple: A tuple containing loss, number of examples, and metrics.
        """
        # For this setup, evaluation is done within the Nextflow pipeline
        # We'll return the metrics from the latest model
        _, _, metrics = self.get_latest_model_from_s3()
        return metrics["loss"], 16709, metrics

    def get_latest_model_from_s3(self):
        """Retrieve the latest model and its information from S3.

        Returns:
            tuple: A tuple containing the model and its information.
        """
        s3 = boto3.client("s3")
        try:
            # List objects in the bucket with the given prefix
            response = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.s3_key_prefix)
            print(f">>> debug: response {response}")

            # Sort objects by last modified date
            sorted_objects = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)

            # Get the latest .h5 and .json files and training_history.csv for n epochs
            latest_h5 = next(obj for obj in sorted_objects if obj["Key"].endswith(".h5"))
            latest_json = next(obj for obj in sorted_objects if obj["Key"].endswith(".json"))
            training_history = os.path.join(
                sorted_objects[0]["Key"].split("trained_models")[0], "logs", "training_history.csv"
            )

            # Download the files
            s3.download_file(self.s3_bucket, latest_h5["Key"], "/tmp/latest_model.h5")
            s3.download_file(self.s3_bucket, latest_json["Key"], "/tmp/latest_model.json")
            s3.download_file(self.s3_bucket, training_history, "/tmp/training_history.csv")

            # Load the model
            model = tf.keras.models.load_model("/tmp/latest_model.h5")

            # Load model info
            with open("/tmp/latest_model.json") as f:
                model_info = json.load(f)

            # Read the CSV file and get the last row as a dictionary
            with open("/tmp/training_history.csv") as file:
                reader = csv.DictReader(file)
                last_row = None
                for row in reader:
                    last_row = row

            metrics = {key: float(value) for key, value in last_row.items()}

            print(f">>> debug: model_info {model_info}")
            print(f">>> debug: metrics {metrics}")
            return model, model_info, metrics

        except ClientError as e:
            print(f"Error getting latest model from S3: {e}")
            return None, None

    def get_parameters(self, model):
        """Extract the parameters from the given model.

        Args:
            model (tf.keras.Model): The model from which to extract parameters.

        Returns:
            list: A list of model parameters.
        """
        return [layer.get_weights() for layer in model.layers]


def client_fn(context: Context):
    """Create a Flower client based on the given context.

    Args:
        context (Context): The context containing node configuration.

    Returns:
        FlowerClient: An instance of FlowerClient.
    """
    print(f">>> debug: context {context}")
    s3_bucket = "eye2gene-main"
    try:
        # Get site from context
        site = context.node_config["site"]

        # Prepare configurations
        s3_key_prefix = f"out/{site}/trained_models/"
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
