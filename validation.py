import random

import pandas as pd

from configs.data_config import data_config
from configs.experiment_config import experiment_config
from dataset.crowdfunding_dataset import CrowdfundingDataset
from model.logistic_regression_model import LogisticRegression
from utils.common_functions import set_seed, write_file
from utils.metrics import average_precision_score

set_seed(experiment_config.seed)


def validate_hyperparams():
    """Makes hyperparameters validation."""
    # Initialize data
    dataset = CrowdfundingDataset(data_config)
    train_data = dataset('train')
    valid_data = dataset('validation')
    test_data = dataset('test')

    # Get validation params
    params_range = experiment_config.validation.params_range
    best_results = {'ap_valid': 0}

    for _iter in range(experiment_config.validation.steps_num):
        random_params = {param: random.choice(params_range[param]) for param in params_range}
        experiment_config.params.update(random_params)

        model = LogisticRegression(experiment_config)
        model.train(train_data['features'], train_data['targets'], valid_data['features'], valid_data['targets'])

        predictions = model.get_model_confidence(train_data['features'])[-1, :]
        ap_train = average_precision_score(train_data['targets'], predictions)

        # Validate
        predictions = model.get_model_confidence(valid_data['features'])[-1, :]
        ap_valid = average_precision_score(valid_data['targets'], predictions)

        print(f'Iteration: {_iter}\t'
              f'Train Average Precision: {ap_train}\t'
              f'Validation Average Precision: {ap_valid}\t'
              f'Parameters: {random_params}\n')

        if ap_valid > best_results['ap_valid']:
            best_results['ap_valid'] = ap_valid
            best_results['params'] = random_params
            best_results['model'] = {'weights': model.weights, 'bias': model.bias}
            print(f'Current best Average Precision for validation set: {ap_valid} (parameters: {random_params})\n')

    write_file(best_results, 'best_results.pickle')

    # Get test predictions with model on the best parameters
    experiment_config.params.update(best_results['params'])
    model = LogisticRegression(experiment_config)
    model.weights = best_results['model']['weights']
    model.bias = best_results['model']['bias']

    test_predictions = model.get_model_confidence(test_data['features'])[-1, :]
    test_results_df = pd.DataFrame({'ID': test_data['path'], 'prediction': test_predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    validate_hyperparams()
