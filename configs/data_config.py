import os

from easydict import EasyDict

from utils.enums import PreprocessingType, LinkageMethod, StoppingCriteria

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

data_config = EasyDict()

# Path to the directory with dataset files
data_config.path_to_data = os.path.join(ROOT_DIR, 'data', 'alzheimer')
data_config.annotation_filename = 'data_info.csv'

data_config.preprocess_type = PreprocessingType.normalization
data_config.preprocess_params = {'a': -1, 'b': 1}  # Or e.g. {'local': False} for PreprocessingType.standardization
data_config.num_classes = 2
data_config.image_size = (128, 128)


data_config.clusterization = EasyDict()
data_config.clusterization.linkage_method = LinkageMethod.single
data_config.clusterization.distance_metric = 'euclidean' # You can also try to implement other distance metrics
data_config.clusterization.stopping_criteria = StoppingCriteria.distance
data_config.clusterization.stopping_criteria_params = {'distance_th': 21.12} # Or e.g. {'clusters_num_min': 2}
