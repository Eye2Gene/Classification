# Eye2Gene

## For Developers

### IDE

It does not matter which IDE you use, though `VSCode` is recommended.
If so, extensions used:

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
- [Mypy](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)
- [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
- [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)

### Setting up

1. Clone the repo

   ```bash
   git clone git@github.com:Eye2Gene/Classification.git
   cd Classification
   ```

2. Set a python env for the project

   Either create a `conda` environment, or any python virtual env solution (e.g. [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/)).

   Be sure to set such python env for the IDE.

3. Install `python-poetry`

   Must be `version >= 2.0`

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   and then run:

   ```bash
   poetry install # see note if using conda
   pre-commit install
   ```

   NOTE: If using `conda`, do `poetry install --only-root` to avoid messing up with our `conda` environment.

4. If using `vscode`, note that this project already have `.vscode/setting.json` defined for `pyproject.toml` and `pre-commit` compatibility.

## Training

### Running

You can train a network by running `train.py` in the `bin/` directory.

It is highly recommended you use a GPU-enabled TensorFlow installation.

Data must be a directory where each folder corresponds to a different class, or can be supplied as a CSV file with columns `file.path` containing a list of filepaths to the relevant images, and `gene` containing the classname of the label of that particular image. In general we recommend the use of CSVs for dataset control, but there is support for both options.

We have provided some example synthetic data, which you can download by running `python bin/download_synthetic_dataset.py`, and some example CSVs pointing to this data.

You can also specify a JSON config file, for which we have provided some commonly used configurations from our experiments.

Example:
`python3 bin/train.py inceptionv3 --epochs 100 --train-dir [DATASET_CSV_TRAIN].csv --val-dir [DATASET_CSV_TEST].csv --model-save-dir trained_models/ --model-log-dir logs/ --cfg configs/36class.json configs/augmentations_baf.json configs/hparam_set_6b.json`

- Using command line

```shell
python bin/train.py inceptionv3 --epochs 3 --train-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_train.csv --val-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_val.csv --cfg configs/63class.json configs/augmentations_baf.json configs/hparam_set_6b_mini.json --gpu 0

# --load-weights-path trained_models/10012025-173806-InceptionV3-3e-32bs-0.0001lr.h5
```

- Using docker

```shell
docker build -t e2g-train .
mkdir -p trained_models
mkdir -p checkpoints
mkdir -p logs
docker run --rm -u $(id -u):$(id -g) --gpus all -v /mnt:/mnt -v $(pwd)/trained_models:/app/trained_models -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/logs:/app/logs -it e2g-train python /app/bin/train.py inceptionv3 --epochs 3 --train-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_train.csv --val-dir /mnt/data1/wwoof/train_csvs/HEX_baf_v4_val.csv --cfg configs/63class.json configs/augmentations_baf.json configs/hparam_set_6b_mini.json --gpu 0
```

```txt
usage: train.py [-h] [--augmentations AUGMENTATIONS] [--batch-size BATCH_SIZE] [--classes CLASSES [CLASSES ...]]
                [--cfg CFG [CFG ...]] [--dataseries-path DATASERIES_PATH] [--dataseries-label DATASERIES_LABEL]
                [--dropout DROPOUT] [--epochs EPOCHS] [--lr LR] [--lr-schedule {linear,poly}] [--lr-power LR_POWER]
                [--load-weights-path LOAD_WEIGHTS_PATH] [--model-save-dir MODEL_SAVE_DIR] [--model-log-dir MODEL_LOG_DIR]
                [--no-weights] [--preview] [--split VALIDATION_SPLIT] [--data-dir DATA_DIR] [--train-dir TRAIN_DIR]
                [--val-dir VAL_DIR] [--workers WORKERS] [--verbose] [--gpu GPU] [--seed SEED]
                model

positional arguments:
  model                 Name of model to train (enter invalid option to list)

options:
  -h, --help            show this help message and exit
  --augmentations AUGMENTATIONS
                        Comma separated values containing augmentations e.g horizontal_flip=True,zoom=0.3
  --batch-size BATCH_SIZE
                        Batch size
  --classes CLASSES [CLASSES ...]
                        List of classes
  --cfg CFG [CFG ...]   Config file to load model config from
  --dataseries-path DATASERIES_PATH
                        Name of dataseries for image paths (if reading from csv)
  --dataseries-label DATASERIES_LABEL
                        Name of dataseries for labels (if reading from csv)
  --dropout DROPOUT     Dropout probability
  --epochs EPOCHS       Number of epochs to train
  --lr LR               Learning rate
  --lr-schedule {linear,poly}
                        Learning rate scheduler
  --lr-power LR_POWER   Power of lr decay, only used when using polynomial learning rate scheduler
  --load-weights-path LOAD_WEIGHTS_PATH
                        Load model weights from file to start training from
  --model-save-dir MODEL_SAVE_DIR
                        Save location for trained models
  --model-log-dir MODEL_LOG_DIR
                        Save location for model logs (used by tensorboard)
  --no-weights          Don't download and use any pretrained model weights, random init
  --preview             Preview a batch of augmented data and exit
  --split VALIDATION_SPLIT
                        Training/Test split (% of data to keep for training, will be halved for validation and testing)
  --data-dir DATA_DIR   Full dataset directory (will be split into train/val/test)
  --train-dir TRAIN_DIR
                        Training data (validation is taken from this)
  --val-dir VAL_DIR     Validation data (can be supplied if you do not want it taken from training data
  --workers WORKERS     Number of workers to use when training (multiprocessing)
  --verbose             Verbose
  --gpu GPU
  --seed SEED           Seed Number
```

## Prediction

There is another script located at `bin/predict.py` which can be given a directory of images (in a structure keras can read), or a CSV file, and a trained model. The script will then output percentages of correct predictions. You can output the results to either `.npy` or to aCSV file, which will copy the input CSV with the accompanying predictions and save it at the specified location (N.B: This only works when an input CSV is given).

The model must be provided as an `.h5` file, then the script will search for a corresponding `.json` file containing the network config.

```txt
usage: predict.py [-h] [--batch-size BATCH_SIZE] image_dir model

positional arguments:
  image_dir
  model

optional arguments:
  --npy                 Location to save a .npy file with the model outputs
  --csv                 Location to save a CSV file with the model outputs
  --embeddings          Location to save a .npy file of the activations of the penultimate model layer
  --no-softmax          Set to output the pre-softmax model predictions instead
```

## Processing

We have included the predictions of Eye2Gene on our internal and external datasets in the `results` folder, as well as a notebook showing how we process these results to get our metrics such as accuracy/AUROC/etc.
