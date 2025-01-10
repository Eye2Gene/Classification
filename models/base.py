"""Model baseclass."""

import copy
import importlib
import json
import os
import time
from collections import Counter
from typing import Any, ClassVar

import keras_cv_attention_models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.summary import create_file_writer as FileWriter

from models import register_model
from models.learning_rates import PolynomialDecay

# import tensorflow as tf
# from tensorflow.compat.v1.keras.backend import get_session
# tf.compat.v1.disable_v2_behavior()


class RegisterModels(type):
    """Metaclass to register models."""

    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict) -> None:
        """Initialize the metaclass with class attributes."""
        super().__init__(name, bases, attrs)
        register_model(name, cls)
        # if not hasattr(cls, 'registered_models'):
        #    cls.registered_models = {}
        # cls.registered_models[name] = cls


class ModelBase(metaclass=RegisterModels):
    """Base class for a Model."""

    config_defaults: ClassVar[dict] = {
        "augmentations": {},
        "batch_size": 32,
        "classes": ["ABCA4", "USH2A"],
        "data_dir": None,
        "dataseries_path": "file_path",
        "dataseries_label": "gene",
        "dropout": 0.0,
        "epochs": 10,
        "input_shape": [256, 256],
        "loss_fn": "categorical_crossentropy",
        "lr": 1e-04,
        "lr_schedule_config": None,
        "lr_power": 1,
        "model_log_dir": "logs",
        "model_save_dir": "trained_models",
        "train_dir": None,
        "val_dir": None,
        "validation_split": 0.2,
        "workers": 8,
        "seed": 123,
    }
    registered_models: ClassVar[dict] = {}

    @classmethod
    def list_models(cls) -> dict:
        """Return a list of all registered models."""
        return cls.registered_models

    def __init__(self, model_config: dict = dict()) -> None:
        """Initialize the model with the given configuration."""
        # Set up variables
        self.name = "base"
        self._config = copy.deepcopy(self.config_defaults)
        self._config.update(model_config)  # Save raw config
        for k, v in self._config.items():
            setattr(self, k, v)

        self.callbacks: list = list()

        # Datagen attrs
        self.train_generator = None
        self.validation_generator = None

        # if not os.path.exists(self.model_save_dir):
        #    raise Exception('Save location {} does not exist'.format(self.model_save_dir))

    def setup(self, set_layers: bool = True) -> None:
        """Set up functions now that child class is declared."""
        for k, v in self._config.items():
            setattr(self, k, v)

        # Set time to obj creation time
        self.train_start = time.strftime("%d%m%Y-%H%M%S")

        if set_layers:
            self.set_layers()
        # self.save_config()

    def set_layers(self) -> None:
        """To be overwritten by child classes."""
        pass

    def compile(self) -> None:
        """Sensible compile defaults."""
        optimizer = Adam(learning_rate=self.lr)  # Change to learning_rate=
        self.model.compile(loss=self.loss_fn, metrics=["accuracy"], optimizer=optimizer)

    def get_generator(
        self, source: str, subset: str = "training", use_split: bool = False, fold: int | None = None, **kwargs: Any
    ) -> ImageDataGenerator:
        """Create and return an ImageDataGenerator based on the given parameters."""
        if hasattr(self, "preprocess_func"):
            mod_name, func_name = self.preprocess_func.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            preprocess = getattr(mod, func_name)
        else:
            preprocess = None

        rescale = 1.0 / 255 if preprocess is None else None

        if subset == "training":
            datagen = ImageDataGenerator(
                **self.augmentations,
                fill_mode="constant",
                rescale=rescale,
                validation_split=self.validation_split,
                preprocessing_function=preprocess,
            )
        elif subset == "validation":
            datagen = ImageDataGenerator(
                fill_mode="constant",
                rescale=rescale,
                validation_split=self.validation_split,
                preprocessing_function=preprocess,
            )
        else:
            raise Exception("Invalid option for subset: {} \n Please supply either 'training' or 'validation'")

        # Train data generator if we have any dataset-wide augmentations that need stats
        if any(["featurewise" in key for key in self.augmentations.keys()]):
            print("Training datagen")
            raise NotImplementedError("Not implemented: Need to implement training for augmentations")

        train_options = dict(
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=self.classes,
            seed=self.seed,
        )
        train_options.update(kwargs)

        if source[-4:] == ".csv":
            import pandas as pd

            csv_data = pd.read_csv(source)

            csv_data[self.dataseries_label] = csv_data[self.dataseries_label].map(str)

            if fold is not None:
                if subset == "training":
                    csv_data = csv_data[csv_data["fold"] != fold]
                elif subset == "validation":
                    csv_data = csv_data[csv_data["fold"] == fold]

            return datagen.flow_from_dataframe(
                dataframe=csv_data,
                x_col=self.dataseries_path,  #'file.path',
                y_col=self.dataseries_label,  #'gene',
                **train_options,
                subset=subset if use_split else None,
            )

        else:
            if fold:
                raise NotImplementedError("'fold' argument incompatable with non-csv files")
            return datagen.flow_from_directory(source, **train_options, subset=subset if use_split else None)

    def set_generators(self) -> None:
        """Set the data generators."""
        # Load data generators
        if self.data_dir:
            # Load data
            self.train_generator = self.get_generator(self.data_dir, "training", use_split=True)
            self.validation_generator = self.get_generator(self.data_dir, "validation", use_split=True)

        elif self.train_dir and self.val_dir:
            self.train_generator = self.get_generator(self.train_dir, "training")
            self.validation_generator = self.get_generator(self.val_dir, "validation")

        else:
            raise Exception("No data dir or train/validation dir")

    def set_callbacks(self, checkpoints: bool = True, tensorboard: bool = True, auto_stopping: bool = False) -> None:
        """Set and return model callbacks based on specified parameters."""
        self.callbacks = list()

        if checkpoints:
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")

            checkpoint = ModelCheckpoint(
                filepath=os.path.join("checkpoints", self.filename()),
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode="max",
            )
            self.callbacks.append(checkpoint)

        if tensorboard:
            log_dir = os.path.join(self.model_log_dir, self.filename()[:-3])
            self.file_writer = FileWriter(os.path.join(log_dir, "metrics"))
            self.file_writer.set_as_default()
            tensorboard_callback = TensorBoard(
                log_dir=log_dir,
                write_graph=True,
                write_images=True,
                histogram_freq=0,
                profile_batch=0,
            )
            self.callbacks.append(tensorboard_callback)

        lr_schedule = None
        config = self.lr_schedule_config
        if config:
            if config.get("lr_schedule") == "polynomial":
                lr_schedule = PolynomialDecay(maxEpochs=self.epochs, initAlpha=self.lr, power=config.get("lr_power"))
            elif config.get("lr_schedule") == "linear":
                lr_schedule = PolynomialDecay(maxEpochs=self.epochs, initAlpha=self.lr, power=1)

        if lr_schedule:
            lr_callback = LearningRateScheduler(lr_schedule)
            self.callbacks.append(lr_callback)

        if auto_stopping:
            es_callback = EarlyStopping(monitor="val_accuracy", mode="max", patience=10)
            self.callbacks.append(es_callback)

    def train(self, workers: int | None = None) -> dict | None:
        """Train the model."""
        if not self.model:
            print("Model not instantiated")
            return None

        if not self.train_generator:
            self.set_generators()

        # Calc class weights
        counter: Counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        # class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
        class_weights = {class_id: max_val / counter.get(class_id, 1.0) for class_id, _ in enumerate(self.classes)}
        print("Class weights:", class_weights)

        if not self.callbacks:
            self.set_callbacks()

        use_multiprocessing = True if workers is not None else False

        # Train
        history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=self.validation_generator,
            class_weight=class_weights,
            steps_per_epoch=(self.train_generator.samples // self.batch_size),
            validation_steps=self.validation_generator.samples // self.batch_size,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
        )

        # Flush the tensorboard events
        self.file_writer.flush()
        self.file_writer.close()

        return history

    def predict(
        self, x_test: Any, return_labels: bool = False, return_filenames: bool = False, **kwargs: Any
    ) -> tuple[Any, ...]:
        """Generate prediction for single image."""
        # print('Predicting not implemented for', self.name)
        if isinstance(x_test, str):
            generator = self.get_generator(x_test, subset="validation", shuffle=False)
            predictions = self.model.predict(generator, verbose=self.verbose, **kwargs)
        else:
            predictions = self.model.predict(x_test, verbose=self.verbose, **kwargs)

        if return_labels:
            true_labels = generator.classes
            if return_filenames:
                return predictions, true_labels, generator.filenames
            else:
                return predictions, true_labels
        else:
            if return_filenames:
                return predictions, generator.filenames
            else:
                return predictions

    def evaluate(self) -> list:
        """Evaluate the model on the validation data."""
        if not self.validation_generator:
            self.set_generators()

        return self.model.evaluate(self.validation_generator, verbose=self.verbose)

    def generate_preview(self) -> None:
        """Generate preview of augmented data."""
        # Clear preview directory
        if not os.path.exists("preview"):
            os.mkdir("preview")
        else:
            for f in os.listdir("preview"):
                os.remove(os.path.join("preview", f))

        source = self.data_dir if self.data_dir else self.train_dir
        i = 0
        for _ in self.get_generator(source, save_to_dir="preview", save_format="jpeg", shuffle=False):
            i += 1
            if i > 10:
                break

    def print_summary(self) -> None:
        """Print the model summary."""
        if self.model:
            self.model.summary()

    def filename(self) -> str:
        """Human readable filename."""
        return f"{self.train_start}-{self.name}-{self.epochs}e-{self.batch_size}bs-{self.lr}lr.h5"

    def save_location(self) -> str:
        """Save location for model."""
        return os.path.join(self.model_save_dir, self.filename())

    def save_config(self) -> None:
        """Save model configuration to disk."""
        # Save training config
        print("self.model_save_dir ", self.model_save_dir)
        print("Saving config to", self.save_location()[:-3] + ".json")
        with open(self.save_location()[:-3] + ".json", "w") as config_file:
            config_file.write(json.dumps(self._config))

    def save(self) -> None:
        """Save Keras model to disk."""
        self.save_config()

        if self.model:
            if self.verbose:
                print("Saving to", self.save_location())

            # Save model and weights
            self.model.save(self.save_location())

    def load(self, model_path: str, update_config: bool = True, set_layers: bool = True) -> "ModelBase":
        """Load model from disk."""
        # TODO: Set this so it prefers supplied config rather than saved config?
        if update_config:
            config_path = model_path[:-3] + ".json"
            with open(config_path) as config_file:
                model_config = json.load(config_file)

            self._config.update(model_config)
        # if self.verbose:
        #    print('Loading model from ', model_path)
        self.model = load_model(model_path)
        # if self.verbose:
        #    print('Model loaded')

        self.setup(set_layers=set_layers)

        return self
