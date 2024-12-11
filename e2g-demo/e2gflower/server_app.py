"""e2gflower: A Flower / E2G app."""

from typing import List, Tuple
import boto3
from botocore.exceptions import ClientError
import tensorflow as tf
import json
import numpy as np

from flwr.common import Metrics, Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_latest_model_from_s3(bucket, prefix):
    s3 = boto3.client('s3')
    try:
        # List objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        # Sort objects by last modified date
        sorted_objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        
        # Get the latest .h5 and .json files
        latest_h5 = next(obj for obj in sorted_objects if obj['Key'].endswith('.h5'))
        latest_json = next(obj for obj in sorted_objects if obj['Key'].endswith('.json'))

        # Download the files
        s3.download_file(bucket, latest_h5['Key'], '/tmp/latest_model.h5')
        s3.download_file(bucket, latest_json['Key'], '/tmp/latest_model.json')

        # Load the model
        model = tf.keras.models.load_model('/tmp/latest_model.h5')

        # Load model info
        with open('/tmp/latest_model.json', 'r') as f:
            model_info = json.load(f)

        return model, model_info
    except ClientError as e:
        print(f"Error loading model from S3: {e}")
        return None, None

def aggregate_models(models):
    # Assuming all models have the same architecture
    aggregated_weights = []
    for layer_weights in zip(*[model.get_weights() for model in models]):
        aggregated_weights.append(np.mean(layer_weights, axis=0))
    return aggregated_weights

def server_fn(context: Context):
    # Load models from both sites
    model_site1, info_site1 = get_latest_model_from_s3("eye2gene-main-site1", "out/1/trained_models/")
    model_site2, info_site2 = get_latest_model_from_s3("eye2gene-main-site2", "out/2/trained_models/")

    # Aggregate models
    if model_site1 and model_site2:
        aggregated_weights = aggregate_models([model_site1, model_site2])
        aggregated_model = tf.keras.models.clone_model(model_site1)
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