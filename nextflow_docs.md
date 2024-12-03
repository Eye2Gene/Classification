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
docker tag eye2gene/e2g-train ghcr.io/eye2gene/e2g-train:v0.0.5

docker push ghcr.io/eye2gene/e2g-train:latest
docker push ghcr.io/eye2gene/e2g-train:v0.0.5
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
  -it e2g-train python /app/bin/train.py inceptionv3 --epochs 10 \
  --train-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_train.csv \
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
N E X T F L O W  ~  version 23.04.3
Launching `main.nf` [lethal_pasteur] DSL2 - revision: 13acb23d5f
executor >  local (1)
[1d/4953ba] process > trainModel [100%] 1 of 1 ✔
Completed at: 28-Oct-2024 12:46:15
Duration    : 9m 57s
CPU hours   : 0.2
Succeeded   : 1
```

## Running nextflow on AWS Batch

```bash
# for [Seqera monitoring](https://cloud.seqera.io)
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

### Check the nodes created

```bash
aws --region eu-west-2 ec2 describe-instances \
--filters "Name=instance-state-name,Values=running" \
--query 'Reservations[*].Instances[*].[InstanceId, InstanceType, PrivateIpAddress, PublicIpAddress]'
```
