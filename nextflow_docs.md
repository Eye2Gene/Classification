# Development

## Release a new Docker build

The pipeline uses docker images on Github Docker Registry. If you make any changes to the underlying files including in the dockerfile, please push them to Dockerhub:

1. Login using a token with access to Github Packages. See [here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) for more info.

```bash
export CR_PAT=YOUR_TOKEN

echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

2. Build docker image and upload to Github Packages

```bash
docker build -t eye2gene/e2g-train .
docker tag eye2gene/e2g-train ghcr.io/eye2gene/e2g-train:latest

docker push ghcr.io/eye2gene/e2g-train:latest
docker push ghcr.io/eye2gene/e2g-train:v0.0.1
```

## Running Docker locally

```bash
docker build -t e2g-train .
mkdir -p trained_models
mkdir -p checkpoints
mkdir -p logs

docker run --rm -u $(id -u):$(id -g) --gpus all \
  -v /mnt:/mnt \
  -v $(pwd)/trained_models:/app/trained_models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -it e2g-train inceptionv3 --epochs 10 \
  -train-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_train.csv \
  --val-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_val.csv \
  --cfg configs/63class.json \
        configs/augmentations_baf.json \
        configs/hparam_set_6b_mini.json \
  --gpu 0

...
Epoch 10: val_accuracy did not improve from 0.33032
## Training complete ##
## Evaluating on test data##
Validation loss: 4.771307945251465
Validation accuracy: 0.21872831881046295
## Model saved ##
```

## Running nextflow locally

> Note: change the docker image inside main.nf to a local image for development.
> Note: set errorStrategy to 'terminate', so that errors are identified faster.

```bash
nextflow run main.nf -c nextflow.config -c local_params.config
```
