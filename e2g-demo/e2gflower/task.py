"""e2gflower: A Flower / E2G app."""

import json
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import boto3
import tensorflow as tf
from botocore.exceptions import ClientError

# Suppress other warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_model(model_path):
    """Load the TensorFlow model from a .h5 file."""
    return tf.keras.models.load_model(model_path)


def get_params(model):
    """Get model parameters as a list of NumPy ndarrays."""
    return [layer.get_weights() for layer in model.layers]


def set_params(model, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    for layer, layer_params in zip(model.layers, parameters):
        layer.set_weights(layer_params)


def load_latest_model_from_s3(bucket, prefix):
    """Load the latest model and its metadata from S3."""
    s3 = boto3.client("s3")
    try:
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        # Sort objects by last modified date
        sorted_objects = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)

        # Get the latest .h5 and .json files
        latest_h5 = next(obj for obj in sorted_objects if obj["Key"].endswith(".h5"))
        latest_json = next(obj for obj in sorted_objects if obj["Key"].endswith(".json"))

        # Download the files
        local_model_path = "/tmp/latest_model.h5"
        local_metadata_path = "/tmp/latest_model.json"
        s3.download_file(bucket, latest_h5["Key"], local_model_path)
        s3.download_file(bucket, latest_json["Key"], local_metadata_path)

        # Load the model
        model = load_model(local_model_path)

        # Load model metadata
        with open(local_metadata_path) as f:
            metadata = json.load(f)

        return model, metadata
    except ClientError as e:
        print(f"Error 'load_latest_model_from_s3' loading model from S3: {e}")
        return None, None


def save_model_to_s3(model, metadata, bucket, key_prefix):
    """Save the model and its metadata to S3."""
    s3 = boto3.client("s3")
    try:
        # Generate unique filename based on timestamp
        timestamp = tf.timestamp().numpy().astype("int64")
        model_filename = f"{timestamp}-model.h5"
        metadata_filename = f"{timestamp}-metadata.json"

        # Save model locally
        local_model_path = f"/tmp/{model_filename}"
        model.save(local_model_path)

        # Save metadata locally
        local_metadata_path = f"/tmp/{metadata_filename}"
        with open(local_metadata_path, "w") as f:
            json.dump(metadata, f)

        # Upload to S3
        s3.upload_file(local_model_path, bucket, f"{key_prefix}/{model_filename}")
        s3.upload_file(local_metadata_path, bucket, f"{key_prefix}/{metadata_filename}")

        # Clean up local files
        os.remove(local_model_path)
        os.remove(local_metadata_path)

        print(f"Model and metadata saved to S3: {bucket}/{key_prefix}/{model_filename}")
    except ClientError as e:
        print(f"Error 'save_model_to_s3' saving model to S3: {e}")


# Note: The following functions are not directly used in this setup,
# but are kept as placeholders in case they're needed for future modifications


def load_data(num_partitions, partition_id, batch_size):
    """Placeholder for data loading function. In the current setup, data loading is handled by the Nextflow pipeline."""
    pass


def train(model, train_data, epochs, device):
    """Placeholder for training function. In the current setup, training is handled by the Nextflow pipeline."""
    pass


def test(model, test_data, device):
    """Placeholder for testing function. In the current setup, evaluation is handled by the Nextflow pipeline."""
    pass
