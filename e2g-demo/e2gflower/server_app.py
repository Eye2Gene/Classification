"""e2gflower: A Flower / E2G app."""

import logging
import os
import time
from typing import Optional, Union

from e2gflower.task import load_latest_model_from_s3, upload_file_to_s3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import numpy as np
from flwr.common import Context, FitRes, Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Suppress other warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class SaveModelStrategy(FedAvg):
    """A strategy that saves the model after each round."""

    def __init__(self, initial_model, model_path_S3, *args, **kwargs):
        """Initialize the SaveModelStrategy.

        Args:
            initial_model (Model): The initial model.
            model_path_S3 (str): The S3 path of the model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.initial_model = initial_model
        self.model_path_S3 = model_path_S3

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average and save the model."""
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays to S3
            print(f"Saving round {server_round} aggregated_ndarrays...")

            tag = time.strftime("%d%m%Y-%H%M%S")
            self.initial_model.set_weights(aggregated_ndarrays)
            self.initial_model.save("temp.h5", include_optimizer=True)
            # e.g. model_path_S3: '03122024-155742-InceptionV3-1e-32bs-0.0001lr.h5'
            new_name = f"avg_r{server_round}_{tag}-{'-'.join(os.path.basename(self.model_path_S3).split('-')[2:])}"
            new_s3_path = os.path.join(os.path.dirname(self.model_path_S3), new_name)

            print(f">>> debug: 'aggregate_fit' new_s3_path {new_s3_path}")
            print(f">>> debug: 'aggregate_fit' temp_h5_size {os.path.getsize('temp.h5') / (1024**2):.2f} MB")

            upload_file_to_s3("temp.h5", new_s3_path, delete=False)

            print(f"Saved aggregated model for round {server_round} to S3: {new_s3_path}")

        return aggregated_parameters, aggregated_metrics


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
    print(f">>> debug: 'def server_fn' context {context}")
    # Load initial model from S3
    initial_model, _metadata, model_path_S3 = load_latest_model_from_s3("eye2gene-main", "initial_model")

    # output messages:
    # /home/ec2-user/e2g-demo/.venv/lib64/python3.9/site-packages/keras/src/optimizers/base_optimizer.py:86:
    # UserWarning: Argument `decay` is no longer supported and will be ignored.
    #   warnings.warn(
    # WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics`
    # will be empty until you train or evaluate the model.
    # WARNING:absl:Error in loading the saved optimizer state. As a result, your model is starting with a freshly
    # initialized optimizer.

    print(f">>> debug: 'def server_fn' models {initial_model}, {_metadata}")

    initial_parameters = ndarrays_to_parameters(initial_model.get_weights())

    # Define the strategy
    strategy = SaveModelStrategy(
        initial_model=initial_model,
        model_path_S3=model_path_S3,
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
