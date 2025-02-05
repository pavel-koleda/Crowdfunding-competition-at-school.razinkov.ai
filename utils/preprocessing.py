import numpy as np

from configs.experiment_config import experiment_config
from utils.enums import PreprocessingType


class ImageDataPreprocessing:
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType, preprocess_params=None):
        self.preprocess_type = preprocess_type

        # A dictionary with the following keys and values:
        #    - {'min': min values, 'max': max values} when preprocess_type is PreprocessingType.normalization
        #    - {'mean': mean values, 'std': std values} when preprocess_type is PreprocessingType.standardization
        self.params = {}

        # Additional parameters initialization
        if isinstance(preprocess_params, dict):
            self.params.update(preprocess_params)

        # Select the preprocess function according to self.preprocess_type
        self.preprocess_func = getattr(self, self.preprocess_type.name)

    def normalization(self, x: np.ndarray, init=False) -> np.ndarray:
        """Transforms x by scaling each pixel to a range [a, b] with self.params['min'] and self.params['max'].

        The general formula is:
            x_normalized = a + (b - a) * (x - x_min) / (x_max - x_min),

            where:
                - a and b are scaling parameters and a = -1, b = 1 to make range [-1, 1]

        Args:
            x: Array of images.
            init: Initialization flag.

        Returns:
            numpy.array: normalized x.
        """
        if init:
            self.params['min'] = np.min(x)
            self.params['max'] = np.max(x)
            self.params['initialized'] = True

        a = self.params.get('a', experiment_config.default.normalization_a)
        b = self.params.get('b', experiment_config.default.normalization_b)

        return a + (b - a) * ((x - self.params['min']) / (self.params['max'] - self.params['min']))

    def standardization(self, x: np.ndarray, init=False) -> np.ndarray:
        """Standardizes x with self.params['mean'] and self.params['std'].

        The general formula is:
            x_standardized = (x - x_mean) / x_std

        Args:
            x: Array of images.
            init: Initialization flag.

        Returns:
            numpy.array: standardized x
        """
        local = self.params.get('local', experiment_config.default.local_standardization)
        if init and not local:
            self.params['mean'] = x.mean()
            self.params['std'] = x.std()
            self.params['initialized'] = True
        elif local:
            self.params['mean'] = x.mean(axis=1, keepdims=True)
            self.params['std'] = x.std(axis=1, keepdims=True)
            self.params['initialized'] = True

        return (x - self.params['mean']) / self.params['std']

    @staticmethod
    def flatten(x: np.ndarray) -> np.ndarray:
        """Reshaping x into a matrix of shape (N, HxW)"""
        return x.reshape(x.shape[0], -1)

    def train(self, x: np.ndarray) -> np.ndarray:
        """Initializes preprocessing function on training data."""
        flattened_x = self.flatten(x)
        return self.preprocess_func(flattened_x, init=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Returns preprocessed data."""
        if not self.params.get('initialized'):
            raise Exception(f"{self.preprocess_type.name} instance is not trained yet. Please call 'train' first.")

        flattened_x = self.flatten(x)
        return self.preprocess_func(flattened_x, init=False)
