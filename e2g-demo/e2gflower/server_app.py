"""e2gflower: A Flower / E2G app."""

import logging
import os

from e2gflower.task import load_latest_model_from_s3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Suppress other warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Calculate the weighted average of accuracy metrics.

    Args:
        metrics (list[tuple[int, Metrics]]): A list of tuples containing the number of examples
        and their corresponding metrics.

    Returns:
        Metrics: A dictionary containing the weighted average accuracy.
    """
    print(f">>> debug: 'def weighted_average' metrics {metrics}")
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Server function to initialize and configure the federated learning server.

    Args:
        context (Context): The context object containing configuration and state information.
    """
    # Load initial model from S3
    initial_model, metadata, _ = load_latest_model_from_s3("eye2gene-main", "initial_model")

    # output messages:
    # /home/ec2-user/e2g-demo/.venv/lib64/python3.9/site-packages/keras/src/optimizers/base_optimizer.py:86:
    # UserWarning: Argument `decay` is no longer supported and will be ignored.
    #   warnings.warn(
    # WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics`
    # will be empty until you train or evaluate the model.
    # WARNING:absl:Error in loading the saved optimizer state. As a result, your model is starting with a freshly
    # initialized optimizer.

    print(f">>> debug: 'def server_fn' models {initial_model}, {metadata}")

    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
