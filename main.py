import os

import pandas as pd

from configs.data_config import data_config
from configs.experiment_config import experiment_config
from dataset.crowdfunding_dataset import CrowdfundingDataset
from model.logistic_regression_model import LogisticRegression
from utils.common_functions import set_seed

set_seed(experiment_config.seed)


def train():
    """Trains logistic regression model."""
    dataset = CrowdfundingDataset(data_config)
    train_data = dataset('train')
    valid_data = dataset('validation')
    # print(train_data)
    # print(train_data['features'].shape)
    model = LogisticRegression(m=train_data['features'].shape[1], experiment_config=experiment_config)
    model.train(train_data['features'], train_data['targets'], valid_data['features'], valid_data['targets'])


def predict():
    """Gets logistic regression model's predictions."""
    dataset = CrowdfundingDataset(data_config)
    test_data = dataset('test')

    model = LogisticRegression(m=test_data['features'][0].shape[1], experiment_config=experiment_config)
    model.load(os.path.join(experiment_config.checkpoints_dir, 'best_checkpoint.pickle'))

    test_predictions = model.get_model_confidence(test_data['features'][0])[-1, :]

    test_results_df = pd.DataFrame({'ID': test_data['features'][1], 'prediction': test_predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    # train()
    predict()
