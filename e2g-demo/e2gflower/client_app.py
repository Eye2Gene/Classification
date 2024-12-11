"""e2gflower: A Flower / E2G app."""

import tensorflow as tf
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp
import boto3
from botocore.exceptions import ClientError
import subprocess
import json
from datetime import datetime

# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, s3_bucket, s3_key_prefix, nextflow_command):
        self.s3_bucket = s3_bucket
        self.s3_key_prefix = s3_key_prefix
        self.nextflow_command = nextflow_command

    def fit(self, parameters, config):
        # Run Nextflow pipeline
        subprocess.run(self.nextflow_command, shell=True, check=True)

        # Get the latest model from S3
        model, model_info = self.get_latest_model_from_s3()
        
        if model is None:
            raise ValueError("No model found in S3 after Nextflow run")

        # Return model parameters, number of examples, and metrics
        return self.get_parameters(model), model_info['num_examples'], model_info['metrics']

    def evaluate(self, parameters, config):
        # For this setup, evaluation is done within the Nextflow pipeline
        # We'll return the metrics from the latest model
        _, model_info = self.get_latest_model_from_s3()
        return model_info['loss'], model_info['num_examples'], model_info['metrics']

    def get_latest_model_from_s3(self):
        s3 = boto3.client('s3')
        try:
            # List objects in the bucket with the given prefix
            response = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.s3_key_prefix)
            
            # Sort objects by last modified date
            sorted_objects = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
            
            # Get the latest .h5 and .json files
            latest_h5 = next(obj for obj in sorted_objects if obj['Key'].endswith('.h5'))
            latest_json = next(obj for obj in sorted_objects if obj['Key'].endswith('.json'))

            # Download the files
            s3.download_file(self.s3_bucket, latest_h5['Key'], '/tmp/latest_model.h5')
            s3.download_file(self.s3_bucket, latest_json['Key'], '/tmp/latest_model.json')

            # Load the model
            model = tf.keras.models.load_model('/tmp/latest_model.h5')

            # Load model info
            with open('/tmp/latest_model.json', 'r') as f:
                model_info = json.load(f)

            return model, model_info

        except ClientError as e:
            print(f"Error getting latest model from S3: {e}")
            return None, None

    def get_parameters(self, model):
        return [layer.get_weights() for layer in model.layers]

def client_fn(context: Context):
    # Determine S3 bucket and key prefix based on the site
    if context.node_config["site"] == "site1":
        s3_bucket = "eye2gene-main-site1"
        s3_key_prefix = "out/1/trained_models/"
        nextflow_profile = "eye2gene_site1"
    elif context.node_config["site"] == "site2":
        s3_bucket = "eye2gene-main-site2"
        s3_key_prefix = "out/2/trained_models/"
        nextflow_profile = "eye2gene_site2"
    else:
        raise ValueError("Invalid site specified")

    # Construct Nextflow command
    nextflow_command = f"nextflow run Eye2Gene/Classification -r main -c aws_params.config -profile {nextflow_profile} -with-tower" # -resume

    return FlowerClient(s3_bucket, s3_key_prefix, nextflow_command).to_client()

app = ClientApp(client_fn=client_fn)