import numpy as np

from utils.enums import PreprocessingType


class Preprocessing:
    """A class for data preprocessing."""

    def __init__(self, preprocess_type: PreprocessingType):
        self.preprocess_type = preprocess_type

        # A dictionary with the following keys and values:
        #    - {'min': min values, 'max': max values} when preprocess_type is PreprocessingType.normalization
        #    - {'mean': mean values, 'std': std values} when preprocess_type is PreprocessingType.standardization
        self.params: dict[str, np.ndarray | bool] = {}

        # Select the preprocess function according to self.preprocess_type
        self.preprocess_func = getattr(self, self.preprocess_type.name)

    def normalization(self, x: np.ndarray, init=False) -> np.ndarray:
        """Transforms x by scaling each feature to a range [-1, 1] with self.params['min'] and self.params['max']

        The general formula is:
            x_normalized = a + (b - a) * (x - x_min) / (x_max - x_min),

            where:
                - a and b are scaling parameters and a = -1, b = 1 to make range [-1, 1]

        Args:
            x: Feature array.
            init: Initialization flag.

        Returns:
            x_normalized (numpy.array)
        """
        if init:
            # TODO: Calculate min and max for each column in x with np.min, np.max, store the values in
            #       self.params['min'] and self.params['max']
            if self.preprocess_type is not PreprocessingType.normalization:
                raise ValueError("PreprocessingType is not normalization")

            self.params['min'] = np.min(x, axis=0)
            self.params['max'] = np.max(x, axis=0)
            # print(self.preprocess_type.name, ' - init')
        # TODO: Implement data normalization using formula from the docstring and min/max values from self.params.
        #           Return the normalized x array
        a, b = -1, 1

        # x_normalized = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min'])
        x_normalized = a + (b - a) * (x - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-8)

        # print(self.preprocess_type.name)
        return x_normalized


    def standardization(self, x: np.ndarray, init=False) -> np.ndarray:
        """Standardizes x with self.params['mean'] and self.params['std']

        The general formula is:
            x_standardized = (x - x_mean) / x_std
        Args:
            x: Feature array.
            init: Initialization flag.

        Returns:
            x_standardized (numpy.array)
        """
        if init:
            # TODO: Calculate mean and std for each column in x with np.mean, np.std, store the values in
            #       self.params['mean'] and self.params['std']
            if self.preprocess_type is not PreprocessingType.standardization:
                raise ValueError("PreprocessingType is not standardization")
            # print(self.preprocess_type.name, ' - init')
            self.params['mean'] = np.mean(x, axis=0)
            self.params['std'] = np.std(x, axis=0)

        # TODO: Implement data standardization using formula from the docstring and mean/std values from self.params.
        #           Return the standardized x array

        # x_standardized = (x - self.params['mean']) / self.params['std']
        x_standardized = (x - self.params['mean']) / (self.params['std'] + 1e-8)
        # print(self.preprocess_type.name)
        return x_standardized


    def identical(self, x: np.ndarray, init=False) -> np.ndarray:
        """Identity function."""
        self.params['identical'] = True
        return x

    def train(self, x: np.ndarray) -> np.ndarray:
        """Initializes preprocessing function on training data."""
        return self.preprocess_func(x, init=True)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Returns preprocessed data."""
        if len(self.params) == 0:
            raise Exception(f"{self.preprocess_type.name} instance is not trained yet. Please call 'train' first.")
        return self.preprocess_func(x, init=False)