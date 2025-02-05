import os
import pickle
import random
from typing import Union

import numpy as np
import pandas as pd


def write_file(file, path: str):
    """Writes file to the given path."""
    extension = os.path.splitext(path)[1]
    if extension == '.pickle':
        with open(path, 'wb') as f:
            pickle.dump(file, f)
    elif extension == '.npy':
        np.save(path, file)
    else:
        print(f'Unknown extension: {extension}')


def read_file(path: str):
    """Reads files."""
    extension = os.path.splitext(path)[1]
    try:
        if extension == '.pickle':
            with open(path, 'rb') as f:
                file = pickle.load(f)
        elif extension == '.npy':
            file = np.load(path)
        else:
            print(f'Unknown extension: {extension}')
            return None
    except FileNotFoundError:
        print(f'File {path} not found')
        return None
    return file


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    """Reads DataFrame file."""
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)
    elif path_to_file.endswith('parquet'):
        return pd.read_parquet(path_to_file)
    else:
        raise ValueError("Unsupported file format")


def set_seed(seed: int):
    """Sets seed for experiments reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
