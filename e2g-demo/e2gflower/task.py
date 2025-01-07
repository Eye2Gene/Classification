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


def load_latest_model_from_s3(bucket: str, prefix: str):
    """Retrieve the latest model files from an S3 bucket.

    Args:
        bucket: The name of the S3 bucket.
        prefix: The prefix path within the S3 bucket.

    Returns:
        A tuple containing the loaded model, its information, and the S3 path of the .h5 file, or (None, None, None)
        if an error occurs.
    """
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

        s3_path_h5 = f"s3://{bucket}/{latest_h5['Key']}"

        # Load the model
        model = load_model(local_model_path)

        # Load model metadata
        with open(local_metadata_path) as f:
            metadata = json.load(f)

        return model, metadata, s3_path_h5
    except ClientError as e:
        print(f"Error 'load_latest_model_from_s3' ClientError loading model from S3: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error load_latest_model_from_s3' Exception loading model from S3: {e}")
        return None, None, None
