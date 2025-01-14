# Development

## Running Docker locally

```bash
docker build -t eye2gene/e2g-train .

DATA_DIR=/mnt/data1/Eye2Gene/classification_params

docker run --rm -u $(id -u):$(id -g) --gpus all \
  -v /mnt:/mnt \
  -v $(pwd)/trained_models:/app/trained_models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  -it eye2gene/e2g-train python /app/bin/train.py inceptionv3 --epochs 10 \
  --train-dir ${DATA_DIR}/HEX_baf_v4_train.csv \
  --val-dir ${DATA_DIR}/HEX_baf_v4_val.csv \
  --cfg ${DATA_DIR}/configs/63class.json \
        ${DATA_DIR}/configs/augmentations_baf.json \
        ${DATA_DIR}/configs/hparam_set_6b_mini.json \
  --gpu 0

  # optional argument:
  --load-weights-path ${DATA_DIR}/test_start_weights/03122024-155742-InceptionV3-1e-32bs-0.0001lr.h5
```

## Running nextflow locally

> Note: change the docker image inside main.nf to a local image for development.

```bash
nextflow run main.nf -c nextflow.config -c local_params.config
```

Or with the latest remote version

```bash
curl -LO https://raw.githubusercontent.com/Eye2Gene/Classification/refs/heads/main/local_params.config
nextflow pull Eye2Gene/Classification
nextflow run Eye2Gene/Classification -r main -c local_params.config
```

## Running nextflow on AWS Batch

```bash
# for [Seqera monitoring / Fusion etc.](https://cloud.seqera.io)
export TOWER_ACCESS_TOKEN=...

# get AWS credentials from https://eye2gene.awsapps.com/start/#
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."

# Run on Eye2Gene Main Account
curl -LO https://raw.githubusercontent.com/Eye2Gene/Classification/refs/heads/main/aws_params.config
nextflow run Eye2Gene/Classification -r main -c aws_params.config -profile eye2gene_main -resume -with-tower
nextflow run Eye2Gene/Classification -r main -c aws_params.config -profile eye2gene_site1 -resume -with-tower
nextflow run Eye2Gene/Classification -r main -c aws_params.config -profile eye2gene_site2 -resume -with-tower
nextflow run Eye2Gene/Classification -r main -c aws_params.config -profile eye2gene_site3 -resume -with-tower
```

## Release a new Docker build

The pipeline uses docker images on Github Docker Registry. If you make any changes to the underlying files including in the dockerfile, please push them to Dockerhub:

1. Login using a token with access to Github Packages. See [here](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) for more info.

   ```bash
   export CR_PAT=YOUR_TOKEN

   echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
   ```

2. Build docker image and upload to Github Packages

```bash
docker build -t eye2gene/e2g-train .;

docker tag eye2gene/e2g-train "ghcr.io/eye2gene/e2g-train:latest";
docker tag eye2gene/e2g-train "ghcr.io/eye2gene/e2g-train:v0.0.10";

docker push "ghcr.io/eye2gene/e2g-train:latest";
docker push "ghcr.io/eye2gene/e2g-train:v0.0.10";
```
