"""e2gflower: A Flower / E2G app."""

import json
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import boto3
import numpy as np
import tensorflow as tf
from botocore.exceptions import ClientError
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Suppress other warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Calculate the weighted average of accuracy metrics.

    Args:
        metrics (list[tuple[int, Metrics]]): A list of tuples containing the number of examples
        and their corresponding metrics.

    Returns:
        Metrics: A dictionary containing the weighted average accuracy.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_latest_model_from_s3(bucket, prefix):
    """Retrieve the latest model files from an S3 bucket.

    Args:
        bucket (str): The name of the S3 bucket.
        prefix (str): The prefix path within the S3 bucket.

    Returns:
        tuple: A tuple containing the loaded model and its information, or (None, None) if an error occurs.
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

        print(f">>> debug: Latest model files: {latest_h5['Key']} and {latest_json['Key']}")

        # Download the files
        s3.download_file(bucket, latest_h5["Key"], "/tmp/latest_model.h5")
        s3.download_file(bucket, latest_json["Key"], "/tmp/latest_model.json")

        # Load the model
        model = tf.keras.models.load_model("/tmp/latest_model.h5")

        # Load model info
        with open("/tmp/latest_model.json") as f:
            model_info = json.load(f)

        return model, model_info
    except ClientError as e:
        print(f"Error ClientError loading model from S3: {e}")
        return None, None
    except Exception as e:
        print(f"Error Exception loading model from S3: {e}")
        return None, None


def aggregate_models(models):
    """Aggregate the weights of multiple models.

    Args:
        models (list): A list of models to aggregate.

    Returns:
        list: A list of aggregated weights.
    """
    # Assuming all models have the same architecture
    aggregated_weights = []
    for layer_weights in zip(*[model.get_weights() for model in models]):
        aggregated_weights.append(np.mean(layer_weights, axis=0))
    return aggregated_weights


def server_fn(context: Context):
    """Server function to initialize and configure the federated learning server.

    Args:
        context (Context): The context object containing configuration and state information.
    """
    models = []
    for nn in range(1, 4):
        model_site, _ = get_latest_model_from_s3("eye2gene-main", f"out/site{nn}/trained_models/")
        models.append(model_site)

    print(f">>> debug: models {models}")

    # Aggregate models
    if all(model is not None for model in models):
        aggregated_weights = aggregate_models(*models)
        aggregated_model = tf.keras.models.clone_model(models[0])
        aggregated_model.set_weights(aggregated_weights)
    else:
        # Use a new model if loading fails
        aggregated_model = tf.keras.applications.InceptionV3(weights=None, classes=6)  # Adjust as needed

    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(aggregated_model.get_weights())

    # Define strategy
    fraction_fit = context.run_config["fraction-fit"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
