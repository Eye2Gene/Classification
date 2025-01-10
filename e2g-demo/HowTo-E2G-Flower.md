# For E2G with Flower

## from Flower/examples/quickstart-MONAI: Run with the Deployment Engine (not online yet)

Using Flower 1.14, at E2G_Main first: `/home/ec2-user/e2g-demo`

```shell
e2g-demo
├── HowTo-E2G-Flower.md
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

- Change `/home/ec2-user/e2g-demo/superexec.Dockerfile` (deleted), `/home/ec2-user/e2g-demo/supernode.Dockerfile` and `/home/ec2-user/e2g-demo/compose.yaml`
- Edit `pyproject.toml`, then

```bash
pip install -e .

docker-compose build
docker-compose up

# copy supernode.Dockerfile to site1 and do:
docker build -f supernode.Dockerfile -t flwr-supernode .
screen -mS SuperNode ~/start-node.sh

# same for site2
docker build -f supernode.Dockerfile -t flwr-supernode .

# then run
flwr run . e2gdemo --stream 2>&1 | tee flwr_run.log
```

---

## move to zsh

```bash
sudo dnf install zsh util-linux-user git -y
sudo chsh -s $(which zsh) ec2-user
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

## Run from remote site

```shell
export TOWER_ACCESS_TOKEN=eyJ0aWQiOiA4ODMxfS5iMmM2MDkyMWRjNGM3NDE0MGYxYjZmNDA0NjZhZGE2MTM4ODk5NjRh # ismail
export TOWER_ACCESS_TOKEN=eyJ0aWQiOiAxMDI0OX0uZWNkZDlhNTQwMmZmODhhYjFmNGY4OGU0MjU4NTI2Nzc2ZjMyYzNlYg== #alan

nextflow pull Eye2Gene/Classification

# site1 @ ~/test_nf_aws_call
nextflow run Eye2Gene/Classification -r main -c ~/.nextflow/assets/Eye2Gene/Classification/aws_params.config -profile eye2gene_site1 -with-tower
# -resume

# site2 @ ~/test_nf_aws_call
nextflow run Eye2Gene/Classification -r main -c ~/.nextflow/assets/Eye2Gene/Classification/aws_params.config -profile eye2gene_site2 -with-tower
# -resume
```

## Instal nvtop on AWS EC2 temp GPU istance

```shell
# access node via AWS EC2 instance Session Manager
bash
alias ltr='ls -ltr'
cd
wget -O nvtop https://github.com/Syllo/nvtop/releases/download/3.1.0/nvtop-x86_64.AppImage
chmod +x nvtop
sudo ./nvtop
```

--- old stuff

```bash
# in e2e_main, ec2-user account: starts SuperLink and SuperExec
screen -mS COMPOSE sudo docker-compose -f /home/ec2-user/flower-demo/compose.yaml up

# in e2e_site1,2,3, ec2-user account:
screen -mS SuperNode ~/start-node.sh

# back to e2e_main, ec2-user account:
flwr run /home/ec2-user/flower-demo/ && sudo docker-compose -f /home/ec2-user/flower-demo/compose.yaml logs -f superexec


# load the docker image
sudo docker load -i flwr-monai-demo-supernode.tar

rsync -avz --progress ec2-user@10.1.134.255:flwr-monai-demo-supernode.tar .
rsync -avz --progress flwr-monai-demo-supernode.tar ec2-user@10.3.129.36:

sudo docker run -it --rm --entrypoint=/bin/bash flwr-demo-supernode:latest

host,10.0.175.85,i-0dc6cd7dd750b482a eu-west-2
1,10.1.134.255,i-0726cae99095f1ffd   eu-west-2
2,10.2.143.97,i-0847fe2474d6582d2    eu-central-1
3,10.3.129.36,i-0f6657fdbb78fab2c    ap-southeast-2

scp -r /mnt/data1/Eye2Gene/classification_params/configs i-0726cae99095f1ffd:
scp -r /mnt/data1/Eye2Gene/classification_params/configs i-0847fe2474d6582d2:

Host site1
        HostName 10.1.134.255
        User ec2-user
        IdentityFile /root/.ssh/site1.pem
Host site2
        HostName 10.2.143.97
        User ec2-user
        IdentityFile /root/.ssh/site2.pem
Host site3
        HostName 10.3.129.36
        User ec2-user
        IdentityFile /root/.ssh/site3.pem

sudo usermod -aG docker ec2-user

### on site1
# install nextflow
# copy aws_params.config main.nf nextflow.config

docker system prune -f; docker volume prune -f; docker image prune -f
```

## use venv

```shell
cd
python3 -m venv .venv
source ~/.venv/bin/activate
pip install --upgrade pip
pip install "flwr==1.14.0" "boto3>=1.26.0" "botocore>=1.29.0" "tensorflow>=2.10.0"

flower-supernode --superlink="10.0.175.85:9092" --insecure --node-config="site='site1'"
flower-supernode --superlink="10.0.175.85:9092" --insecure --node-config="site='site2'"
```

4326 in HEX_baf_v4_val.csv
16709 in HEX_baf_v4_train.csv

```shell
# run in main
export TF_CPP_MIN_LOG_LEVEL=3
flower-superlink --insecure

# then run for each client site
./start-node.sh

# then in main again, run
export TF_CPP_MIN_LOG_LEVEL=3
flwr run . e2gdemo --stream 2>&1 | tee flwr_run.log
```
