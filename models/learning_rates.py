"""This module contains classes for implementing different learning rate decay schedules."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow


class LearningRateDecay:
    """Base class for learning rate decay schedules."""

    def __call__(self, epoch: int) -> float:
        """Compute the learning rate for a given epoch."""
        raise NotImplementedError("Subclasses must implement this method")

    def plot(self, epochs: int, title: str = "Learning Rate Schedule") -> None:
        """Plot the learning rate schedule over the specified number of epochs."""
        # compute the set of learning rates for each corresponding epoch
        epoch_range = range(epochs)
        lrs = [self(i) for i in epoch_range]

        # plot the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epoch_range, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
    """Implement a step-based decay schedule."""

    def __init__(self, initAlpha: float = 0.01, factor: float = 0.25, dropEvery: int = 10):
        """Initialize StepDecay with initial learning rate, decay factor, and drop interval."""
        self.initAlpha: float = initAlpha
        self.factor: float = factor
        self.dropEvery: int = dropEvery

    def __call__(self, epoch: int) -> float:
        """Compute the learning rate for the current epoch."""
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor**exp)
        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    """Implement a polynomial decay schedule."""

    def __init__(self, maxEpochs: int = 100, initAlpha: float = 0.01, power: float = 1.0) -> None:
        """Initialize PolynomialDecay with maximum number of epochs, initial learning rate, and power."""
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch: int) -> float:
        """Compute the learning rate for the current epoch."""
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        # return the new learning rate
        tensorflow.summary.scalar("learning rate", data=float(alpha), step=epoch)
        return float(alpha)
