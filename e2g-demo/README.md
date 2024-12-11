---
tags: [quickstart, medical, vision]
dataset: [MedNIST]
framework: [MONAI]
---

## [Note -  This is inpired by Flower/examples/quickstart-MONAI]
The quickstart-monai example can be found [here](https://github.com/adap/flower/tree/main/examples/quickstart-monai).

This directory has the following structure: 

```shell
e2g-demo
├── e2gflower
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
├── compose.yaml         # to create superexec and superlink 
├── superexec.Dockerfile # to build the superexec in deployment mode
├── supernode.Dockerfile # to build the supernode in deployment mode 
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `e2gflower` package.

```bash
pip install -e .
```

### Run with the Deployment Engine

Running this example, as in shown demo, assumes that you have already installed all dependencies (see above) and that this repository is cloned. We also assume that docker is installed on your machine. 

We first build the superExec and superLink which are defined in the `compose.yaml`: 
```bash
docker compose build
```
while in this repository. We should be able to see that the images are built. We can thereafter call:
```bash 
docker compose up 

# To follow superExec logs 
docker compose logs -f superexec

# To follow superLink logs
docker compose logs -f superlink
``` 
to ensure that they are created and running. __Important__ is to note down the name of the network which we will use: 
```bash 
docker network ls
``` 
The network name should be `flwr-<COMPANY>-demo_default`. 

In order to run a federated example, we need to build at least 2 supernode images. We can do that in other terminal windows. 

```bash 
# Build superNode 1 
docker build -f supernode.Dockerfile -t flwr-supernode-1
```
```bash
# Build superNode 2
docker build -f supernode.Dockerfile -t flwr-supernode-2
```

Our federation is idle, and the superExec is waiting for us to start a run. We connect the superNodes to the federation: 
```bash 
docker run --rm --network flwr-<COMPANY>-demo_default flwr-supernode-1:latest --superlink="superlink:9092" --node-config "partition-id=0 num-partitions=10" --insecure --isolation=subprocess
```

We thereafter verify that in `pyproject.toml` we see that the last lines are set to `[tool.flwr.federations.<FEDERATION>]`
`address = "localhost:9093"`
`insecure = true`.

Our federated example is called `<FEDERATION>`, our server address is our current machine `localhost:9093`, and for demo purposes are communication is set to `insecure`. If you are running this example on a remote server, change the `localhost` to your remote IP address.

We can thereafter start our run: 
```bash 
flwr run . <FEDERATION>
```
and we should see the training start. We can verify that the training has started if we see progress in the superexec logs as in CLI output below: 
```bash 
superexec-1  | INFO :      [INIT]
superexec-1  | INFO :      Using initial global parameters provided by strategy
superexec-1  | INFO :      Starting evaluation of initial global parameters
superexec-1  | INFO :      Evaluation returned no results (`None`)
superexec-1  | INFO :      
superexec-1  | INFO :      [ROUND 1]
superexec-1  | INFO :      configure_fit: strategy sampled 2 clients (out of 2)
superexec-1  | INFO :      configure_fit: strategy sampled 2 clients (out of 2)
superexec-1  | INFO :      aggregate_fit: received 2 results and 0 failures
superexec-1  | WARNING :   No fit_metrics_aggregation_fn provided
superexec-1  | INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
superexec-1  | INFO :      aggregate_fit: received 2 results and 0 failures
superexec-1  | WARNING :   No fit_metrics_aggregation_fn provided
superexec-1  | INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
superexec-1  | INFO :      aggregate_evaluate: received 2 results and 0 failures
```
