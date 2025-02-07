import os

from easydict import EasyDict

from utils.enums import PreprocessingType, LinkageMethod, StoppingCriteria

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_config = EasyDict()

# Path to the directory with dataset files

data_config.path_to_data = os.path.join(ROOT_DIR, 'data')
data_config.type = {
    'train': 'crowdfunding_data_train.parquet',  # Training set
    'validation': 'crowdfunding_data_valid.parquet',  # Valid set
    'test': 'crowdfunding_data_test.parquet'  # Test set
}

data_config.preprocess_type = PreprocessingType.standardization
data_config.preprocess_params = {'local': False}  # Or e.g. {'local': False} for PreprocessingType.standardization
data_config.num_classes = 2


data_config.categories = {
    'RatecodeID': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0],
    'pickup_day_of_week': [0, 1, 2, 3, 4, 5, 6],
}


data_config.clusterization = EasyDict()
data_config.clusterization.linkage_method = LinkageMethod.single
data_config.clusterization.distance_metric = 'euclidean' # You can also try to implement other distance metrics
data_config.clusterization.stopping_criteria = StoppingCriteria.distance
data_config.clusterization.stopping_criteria_params = {'distance_th': 21.12} # Or e.g. {'clusters_num_min': 2}
