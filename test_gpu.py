#!/usr/bin/env python3
# ruff: noqa: D103, D100
import multiprocessing
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def check_tensorflow_gpu() -> None:
    try:
        import tensorflow as tf

        # Check if TensorFlow is using GPU
        if tf.config.list_physical_devices("GPU"):
            print(">>> TensorFlow is using GPU")
        else:
            print(">>> TensorFlow is not using GPU")
    except Exception:
        print("ERROR: TensorFlow failed to load")


def check_pytorch_gpu() -> None:
    try:
        import torch

        # Check if PyTorch is using GPU
        if torch.cuda.is_available():
            print(">>> PyTorch is using GPU")
            print(f"Available cuda = {torch.cuda.is_available()}")
            print(f"GPUs availables = {torch.cuda.device_count()}")
            print(f"Current device = {torch.cuda.current_device()}")
            print(f"Current Device location = {torch.cuda.device(0)}")
            print(f"Name of the device = {torch.cuda.get_device_name(0)}")
        else:
            print(">>> PyTorch is not using GPU")
    except Exception:
        print("ERROR: PyTorch failed to load")


if __name__ == "__main__":
    # Create two child processes to call the functions
    p1 = multiprocessing.Process(target=check_tensorflow_gpu)
    p2 = multiprocessing.Process(target=check_pytorch_gpu)

    # Start the processes
    p1.start()
    p2.start()

    # Wait for the processes to finish
    p1.join()
    p2.join()
