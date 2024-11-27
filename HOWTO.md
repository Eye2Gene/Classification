# How To

For Ismail.

I'm basically trying to run nextflow in site1 (and then site2), reproducing the way it does from `Main`.
The idea is that if `flwr` is basically running the model remotely, using the remote data, then
site1 and site2 should be able to run when executed directly.
Using half of the train and validation images (2 partitions, one for each site).

```bash
# For site1/site2
# copy Access keys from https://eye2gene.awsapps.com/start for site1 and site2 respectively
export AWS_REGION=eu-west-2 # for site1
export AWS_REGION=eu-central-1 # for site2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...
code --remote ssh-remote+i-0726cae99095f1ffd /home/ec2-user/ # vscode to site1
# see /home/ec2-user/README.md
```

## Map

```csv
host,10.0.175.85,i-0dc6cd7dd750b482a,eu-west-2
1,10.1.134.255,i-0726cae99095f1ffd,eu-west-2
2,10.2.143.97,i-0847fe2474d6582d2,eu-central-1
3,10.3.129.36,i-0f6657fdbb78fab2c,ap-southeast-2
```

----- Ignore bellow -----

## To install `nvtop`

```bash
    wget https://github.com/Syllo/nvtop/releases/download/3.1.0/nvtop-x86_64.AppImage
```
