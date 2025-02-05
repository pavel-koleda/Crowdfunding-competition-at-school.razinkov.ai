import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.common_functions import read_dataframe_file
from utils.enums import SetType
from utils.preprocessing import ImageDataPreprocessing


class AlzheimerDataset:
    """A class for the Alzheimer dataset. This class reads the data and preprocesses it."""

    def __init__(self, config):
        """Initializes the Alzheimer dataset class instance."""
        self.config = config

        # Read an annotation file that contains the image path, set_type, and target values for the entire dataset
        self.annotation = read_dataframe_file(os.path.join(config.path_to_data, config.annotation_filename))

        # Preprocessing class initialization
        self.preprocessing = ImageDataPreprocessing(config.preprocess_type, config.preprocess_params)

        # Read and preprocess the data
        self.data = {}
        for set_type in SetType:
            self.data[set_type.name] = self.preprocess_data(set_type)

    def preprocess_data(self, set_type: SetType) -> dict:
        """Reads and preprocesses the data.

        Args:
            set_type: Data set_type from SetType.

        Returns:
            A dict with the following data:
                {'features': images (numpy.ndarray), 'targets': targets (numpy.ndarray), 'paths': list of paths}
        """
        annotation = self.annotation[self.annotation['set'] == set_type.name]
        tqdm_description = f'{set_type.name.title()} set images reading'
        images = []

        for i, row in tqdm(annotation.iterrows(), total=len(annotation), desc=tqdm_description):
            image = cv2.imread(os.path.join(self.config.path_to_data, row['path']), cv2.IMREAD_GRAYSCALE)
            if image.shape != self.config.image_size:
                image = cv2.resize(image, self.config.image_size, interpolation=cv2.INTER_LANCZOS4)
            images.append(image)

        images = np.stack(images, dtype=np.float64)

        if set_type is SetType.train:
            images = self.preprocessing.train(images)
        else:
            images = self.preprocessing(images)

        if set_type is not SetType.test:
            targets = annotation['target'].to_numpy(dtype=int)
        else:
            targets = None

        return {'features': images, 'targets': targets, 'path': annotation['path'].values}

    def __call__(self, set_type: str) -> dict:
        """Returns preprocessed data."""
        return self.data[set_type]
