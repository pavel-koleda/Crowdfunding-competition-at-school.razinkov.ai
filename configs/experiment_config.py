import os

import numpy as np
from easydict import EasyDict

from utils.enums import LossType, RegularizationType
from utils.enums import WeightsInitType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_config = EasyDict()
experiment_config.seed = 0
experiment_config.loss_type = LossType.softmax
experiment_config.loss_name = experiment_config.loss_type.name.title()
experiment_config.metric_name = 'AP'
experiment_config.experiment_name = 'logistic_regression'

experiment_config.logs_dir = os.path.join(ROOT_DIR, 'experiment_logs')
experiment_config.params_dir = os.path.join(experiment_config.logs_dir, experiment_config.experiment_name, 'params')
experiment_config.plots_dir = os.path.join(experiment_config.logs_dir, experiment_config.experiment_name, 'plots')
experiment_config.checkpoints_dir = os.path.join(
    experiment_config.logs_dir, experiment_config.experiment_name, 'checkpoints'
)

experiment_config.save_model_iter = 100
experiment_config.load_model = False
experiment_config.load_model_epoch = None
experiment_config.load_model_path = os.path.join(
    experiment_config.checkpoints_dir, f'checkpoint_{experiment_config.load_model_epoch}.pickle'
)
experiment_config.early_stopping = {'min_delta': 1e-5, 'patience': 500}  # Set to None if no need in early stopping
experiment_config.params = {
    'learning_rate': 1e-3, 'num_iterations': 1000, 'reg_coefficient_ridge': 0.01, 'reg_coefficient_lasso': 0.01,
    'reg_type': RegularizationType.ridge.value
}

# Default parameters
experiment_config.default = EasyDict()
experiment_config.default.normalization_a = -1
experiment_config.default.normalization_b = 1
experiment_config.default.local_standardization = False
experiment_config.default.sigma = 1
experiment_config.default.uniform_epsilon = 1

# Weights initialization
experiment_config.weights_initialization = EasyDict()
experiment_config.weights_initialization.type = WeightsInitType.normal
experiment_config.weights_initialization.kwargs = {'sigma': 0.01}  # Or e.g. {'epsilon': 1} for WeightsInitType.uniform
experiment_config.weights_initialization.zero_bias = True

# Validation
experiment_config.validation = EasyDict()
experiment_config.validation.samples_num = 40
experiment_config.validation.steps_num = 20
experiment_config.validation.params_range = {
    'learning_rate': np.logspace(np.log10(1e-6), np.log10(1e-3), experiment_config.validation.samples_num),
    'reg_coefficient_lasso': np.logspace(np.log10(1e-6), np.log10(1e-3), experiment_config.validation.samples_num),
    'reg_coefficient_ridge': np.logspace(np.log10(1e-6), np.log10(1e-3), experiment_config.validation.samples_num),
    'reg_type': [
        RegularizationType.ridge.value, RegularizationType.lasso.value, RegularizationType.none.value,
        RegularizationType.lasso_ridge.value
    ],
}

os.makedirs(experiment_config.logs_dir, exist_ok=True)
os.makedirs(experiment_config.checkpoints_dir, exist_ok=True)
os.makedirs(experiment_config.params_dir, exist_ok=True)
os.makedirs(experiment_config.plots_dir, exist_ok=True)
