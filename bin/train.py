"""
Train various different architectures of neural network using Keras
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(1, os.path.join(sys.path[0], ".."))


def parse_augs(augs):
    if not augs:
        return dict()

    pairs = augs.split(",")
    parsed_augs = dict()
    for setting in pairs:
        var, val = setting.split("=")

        # Integer/Float parsing
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        # Bool parsing
        if str(val).lower() == "true":
            val = True
        elif str(val).lower() == "false":
            val = False

        parsed_augs[var] = val
    return parsed_augs


def get_csv_row_count(file_path):
    df = pd.read_csv(file_path)
    return len(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", help="Name of model to train (enter invalid option to list)"
    )
    parser.add_argument(
        "--augmentations",
        help="Comma separated values containing augmentations e.g horitzontal_flip=True,zoom=0.3",
    )
    parser.add_argument("--batch-size", help="Batch size", type=int)
    parser.add_argument("--classes", help="List of classes", nargs="+")
    parser.add_argument(
        "--cfg", help="Config file to load model config from", nargs="+"
    )
    parser.add_argument(
        "--dataseries-path",
        help="Name of dataseries for image paths (if reading from csv)",
    )
    parser.add_argument(
        "--dataseries-label", help="Name of dataseries for labels (if reading from csv)"
    )
    parser.add_argument("--dropout", type=float, help="Dropout probability")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--lr", help="Learning rate", type=float)
    parser.add_argument(
        "--lr-schedule", choices=["linear", "poly"], help="Learning rate scheduler"
    )
    parser.add_argument(
        "--lr-power",
        type=int,
        help="Power of lr decay, only used when using polynomial learning rate scheduler",
        default=1,
    )
    parser.add_argument(
        "--resume-from",
        help="Load model weights from file to start training from",
    )
    parser.add_argument(
        "--model-save-dir",
        default="trained_models",
        help="Save location for trained models",
    )
    parser.add_argument(
        "--model-log-dir",
        default="logs",
        help="Save location for model logs (used by tensorboard)",
    )
    parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Don't download and use any pretrained model weights, random init",
    )
    parser.add_argument(
        "--npy",
        action="store_true",
        help="Use numpy object for saving instead of .h5 - WARNING: not inter-operable with predict script.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview a batch of augmented data and exit",
    )
    parser.add_argument(
        "--split",
        dest="validation_split",
        help="Training/Test split (%% of data to keep for training, will be halved for validation and testing)",
        type=float,
    )
    parser.add_argument(
        "--data-dir", help="Full dataset directory (will be split into train/val/test)"
    )
    parser.add_argument(
        "--train-dir", help="Training data (validation is taken from this)"
    )
    parser.add_argument(
        "--val-dir",
        help="Validation data (can be supplied if you do not want it taken from training data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use when training (multiprocessing)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, help="Seed Number")

    args = parser.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # TODO: Move to model description?
    defaults = {
        "augmentations": {},
        "batch_size": 32,
        "classes": ["ABCA4", "USH2A"],
        "data_dir": None,
        "dataseries_path": "file_path",
        "dataseries_label": "gene",
        "dropout": 0.0,
        "epochs": 10,
        "input_shape": [256, 256],
        "lr": 1e-04,
        "lr_schedule_config": None,
        "lr_power": 1,
        "model_log_dir": "logs",
        "model_save_dir": "trained_models",
        "train_dir": None,
        "val_dir": None,
        "validation_split": 0.2,
        "workers": 8,
        "seed": random.randint(0, 2**32 - 1),
    }

    model_config = defaults.copy()

    # Overwrite the above defaults with any config files
    if args.cfg:
        for cfg_file in args.cfg:
            with open(cfg_file, "r") as f:
                cfg = json.load(f)
            model_config.update(cfg)

    # Overwrite the above defaults with the command line arguments
    no_arg = ["augemntations", "cfg", "model", "no-weights", "preview"]
    arg_dict = vars(args)
    for k, v in arg_dict.items():
        if not (k in no_arg or v is None):
            model_config[k] = v

    # Manually parse remaining arguments
    if args.model:
        model_config["model_name"] = args.model
    model_config["use_imagenet_weights"] = not (args.no_weights or args.resume_from)

    # Parse lr schedule
    if args.lr_schedule == "poly":
        model_config["lr_schedule_config"] = {
            "lr_schedule": "polynomial",
            "initial_lr": model_config.get("lr"),
            "lr_power": args.lr_power,
        }
    elif args.lr_schedule == "linear":
        model_config["lr_schedule_config"] = {
            "lr_schedule": "linear",
            "inital_lr": model_config.get("lr"),
            "lr_power": 1,
        }

    # Parse augmentations
    if args.augmentations:
        try:
            model_config["augmentations"].extend(parse_augs(args.augmentations))
        except Exception as e:
            print(
                "Error parsing augmentations, make sure it is in csv format, with each value being setting=value"
            )
            print(e)
            exit(1)

    # if not args.data_dir:
    #    if not args.train_dir:
    #        print('Need to supply --train-dir')
    #        sys.exit(1)

    # Set tf to grow into GPU memory, not pre-allocate
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)

    if len(gpus) == 0:
        print("No GPUs found!")
        sys.exit(1)

    if args.verbose:
        print("GPUs: ", gpus)
        print(model_config)

    # Create model

    """
    if model_config['model_name'] == 'vgg16':
        model = VGG16(model_config)
    elif model_config['model_name'] == 'inception_resnetv2':
        model = InceptionResnetV2(model_config)
    elif model_config['model_name'] == 'inceptionv3':
        model = InceptionV3(model_config)
    elif model_config['model_name'] == 'custom':
        model = Custom(model_config)
    elif model_config['model_name'] == 'nasnetlarge':
        model = NASNetLarge(model_config)
    """

    from models import get_model, list_models

    modelcls = get_model(model_config["model_name"])
    if modelcls:
        model = modelcls(model_config)
    else:
        print("Unknown/No model selected!")
        print("\nAvailable models:")
        for m in list_models():
            print("-", m)
        sys.exit(1)

    if args.resume_from and os.path.exists(args.resume_from):
        weight_path = args.resume_from
        if weight_path.split(".")[-1] == 'npy':
            model_weights = np.load(weight_path, allow_pickle=True)
            model.model.set_weights(model_weights)
        else:
            model.load(args.resume_from, update_config=False, set_layers=False)
        print(f"Loaded weights from {args.resume_from}")

    if args.verbose:
        model.print_summary()

    if args.preview:
        print("## Generating preview data ##")
        model.generate_preview()
        sys.exit(1)

    os.makedirs(model_config["model_save_dir"], exist_ok=True)

    # Training
    model.compile()

    print("## Training on train data ##")
    model.save_config()
    history = model.train(workers=model_config["workers"])

    # Save training history
    os.makedirs(model_config["model_log_dir"], exist_ok=True)
    history_file = os.path.join(model_config["model_log_dir"], "training_history.csv")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_file, index=False)
    print(f"Training history saved to {history_file}")

    # Update model._config with the last key-values of history.history
    for key, value in history.history.items():
        model._config[key] = value[-1]

    # Add train_size and val_size to model._config
    model._config["train_size"] = get_csv_row_count(model._config["train_dir"])
    model._config["val_size"] = get_csv_row_count(model._config["val_dir"])

    print("## Training complete ##")

    print("## Evaluating on test data ##")
    score = model.evaluate()
    print("Validation loss:", score[0])
    print("Validation accuracy:", score[1])
    model.accuracy = np.round(score[1] * 100)

    model.save(npy=args.npy)
    print("## Model saved ##")

    # print('## Predicting ##')
    # model.test_generator.reset()
    # predictions = model.predict(args.test_data)
