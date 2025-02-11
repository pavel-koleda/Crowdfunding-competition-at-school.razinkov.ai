import os
from typing import Union

import numpy as np
from tqdm import tqdm

from configs.data_config import data_config
from utils.common_functions import write_file, read_file
from utils.enums import SetType, LoggingParamType, LossType, RegularizationType
from utils.metrics import average_precision_score
from utils.params_logger import ParamsLogger


class LogisticRegression:
    """A class for implementing and training a logistic regression model using the numpy library."""

    def __init__(self, m: int, experiment_config):
        self.experiment_config = experiment_config
        self.input_vector_dimension = m
        # Params initialization
        self._init_params()

        # Weights and bias initialization
        weights_initialization_func_name = f'_init_weights_{self.experiment_config.weights_initialization.type.name}'
        getattr(self, weights_initialization_func_name)(**self.experiment_config.weights_initialization.kwargs)

        # Params logger initialization
        self.params_logger = ParamsLogger(experiment_config)
        self.start_iteration = self.prepare_model()

    def _init_params(self):
        """Initializes params needed to train model according to the experiment configuration."""
        if self.experiment_config.loss_type.value == LossType.sigmoid.value:
            self.activation_function = self.__sigmoid
            self.output_vector_dimension = 1
        else:
            self.activation_function = self.__softmax
            self.output_vector_dimension = data_config.num_classes

        
        for param_name, param in self.experiment_config.params.items():
            setattr(self, param_name, param)

    def _init_weights(self, weights, bias):
        """Fills the weights and the bias with the provided values."""
        self.weights = weights
        self.bias = bias

    def _init_weights_normal(self, sigma=None):
        """Fills the weights and the bias with random values using a normal distribution.

        W ~ N(0, sigma^2),

        where:
            - sigma can be defined in self.experiment_config.weights_initialization.kwargs
        """
        sigma = sigma if sigma is not None else self.experiment_config.default.sigma
        weights_size = (self.output_vector_dimension, self.input_vector_dimension)
        self.weights = np.random.normal(0, sigma, size=weights_size)

        if self.experiment_config.weights_initialization.zero_bias:
            self.bias = np.zeros((self.output_vector_dimension, 1))
        else:
            self.bias = np.random.normal(0, sigma, size=(self.output_vector_dimension, 1))

    def _init_weights_uniform(self, epsilon=None):
        """Fills the weights and the bias with random values using a uniform distribution.

        W ~ U(-epsilon, epsilon),

        where:
            - epsilon can be defined in self.experiment_config.weights_initialization.kwargs
        """
        epsilon = epsilon if epsilon is not None else self.experiment_config.default.uniform_epsilon
        weights_size = (self.output_vector_dimension, self.input_vector_dimension)
        self.weights = np.random.uniform(-epsilon, epsilon, size=weights_size)

        if self.experiment_config.weights_initialization.zero_bias:
            self.bias = np.zeros((self.output_vector_dimension, 1))
        else:
            self.bias = np.random.uniform(-epsilon, epsilon, size=(self.output_vector_dimension, 1))

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        """Computes the model output by applying a linear transformation to the input data.

        The linear transformation is defined by the equation:
            z = W * x + b

            where:
                - W (a KxD matrix) represents the weight matrix,
                - x (a DxN matrix, also known as 'inputs') represents the input data,
                - b (a vector of length K) represents the bias vector,
                - z represents the model output before activation.
        Args:
            inputs: NxD matrix.

        Returns:
            np.ndarray: The model output before activation, KxN matrix.
        """
        return self.weights @ inputs.T + self.bias

    @staticmethod
    def __softmax(model_output: np.ndarray) -> np.ndarray:
        """Computes the softmax function on the model output.

        The formula for softmax function is:
            y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

            where:
                - y_j is the softmax probability of class j,
                - z_j is the model output for class j before softmax,
                - K is the total number of classes,
                - Σ denotes summation.

        For numerical stability, subtract the max value of model_output before exponentiation:
            z_j = z_j - max(model_output) (max by rows)

        Args:
            model_output: The model output before softmax, KxN matrix.

        Returns:
            np.ndarray: The softmax probabilities, KxN matrix.
        """
        stable_model_output = model_output - np.max(model_output, axis=0)
        exp_model_output = np.exp(stable_model_output)
        return exp_model_output / np.sum(exp_model_output, axis=0)

    @staticmethod
    def __sigmoid(model_output: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function on the model output.

        The formula for sigmoid function is:
            y = 1 / (1 + e^(-z))

            where:
                - y is the sigmoid probability,
                - z is the model output before sigmoid,

        For numerical stability, for negative values compute sigmoid using this function:
            y = e^(z) / (1 + e^(z))

        Args:
            model_output: The model output before sigmoid, N vector.

        Returns:
            np.ndarray: The sigmoid probabilities, N vector.
        """
        z = np.zeros_like(model_output)
        positive_outputs = model_output >= 0
        z[positive_outputs] = 1 / (1 + np.exp(-model_output[positive_outputs]))

        exp_negative = np.exp(model_output[~positive_outputs])
        z[~positive_outputs] = exp_negative / (1 + exp_negative)
        return z

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """Calculates the model confidence.

        Model confidence is represented as:
            y(x, b, W) = Activation(Wx + b) = Activation(z)

            where:
                - W (a KxD matrix) represents the weight matrix,
                - x (a DxN matrix, also known as 'inputs') represents the input data,
                - b (a vector of length K) represents the bias vector,
                - z represents the model output before activation,
                - y represents the model output after activation.

        Args:
            inputs: NxD matrix.

        Returns:
            np.ndarray: The model output after activation, KxN matrix.
        """
        z = self.__get_model_output(inputs)
        y = self.activation_function(z)
        return y

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the cost function with respect to the weights.

        The gradient of the error with respect to the weights (∆w E) can be computed using the formula:
            ∇w E = (1 / N) * (y - t) * x^T,

            where:
                - y represents the model output after activation,
                - t is for the vectors of target values (one-hot encoded in case of softmax activation),
                - x (a DxN matrix, also known as 'inputs') represents the input data,
                - N is the number of data points.

        For L1-regularization (Lasso):
            ∇w E = |(1 / N) * (y - t) * x^T + λ, if w > 0
                   |(1 / N) * (y - t) * x^T, if w = 0
                   |(1 / N) * (y - t) * x^T - λ, if w < 0

        For L2-regularization (Ridge):
            ∇w E = (1 / N) * (y - t) * x^T + λ * w

        For L1 & L2 regularization:
            ∇w E = |(1 / N) * (y - t) * x^T + λ * w + λ, if w > 0
                   |(1 / N) * (y - t) * x^T + λ * w, if w = 0
                   |(1 / N) * (y - t) * x^T + λ * w - λ, if w < 0

        Args:
            inputs: NxD matrix.
            targets: NxK matrix.
            model_confidence: KxN matrix.

        Returns:
             np.ndarray: KxD matrix.
        """
        if self.reg_type == RegularizationType.none.value:
            regularization_term = 0
        elif self.reg_type == RegularizationType.ridge.value:
            regularization_term = self.reg_coefficient_ridge * self.weights
        elif self.reg_type == RegularizationType.lasso.value:
            regularization_term = self.reg_coefficient_lasso * np.sign(self.weights)
        elif self.reg_type == RegularizationType.lasso_ridge.value:
            regularization_term = self.reg_coefficient_ridge * self.weights \
                                  + self.reg_coefficient_lasso * np.sign(self.weights)
        else:
            raise Exception('Unknown regularization type')
        return (model_confidence - targets.T) @ inputs / targets.shape[0] + regularization_term

    @staticmethod
    def __get_gradient_b(targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the cost function with respect to the bias.

        The gradient of the error with respect to the bias (∇b E) can be computed using the formula:
            ∇b E = (1 / N) * Σ(i=0 to N-1) (y_i - t_i)

            where:
                - y represents the model output after activation,
                - t is for the vectors of target values (one-hot encoded in case of softmax activation),
                - N is the number of data points.
        Args:
            targets: NxK matrix.
            model_confidence: KxN matrix.

        Returns:
             np.ndarray: Kx1 matrix.
        """
        return (model_confidence - targets.T).mean(axis=1, keepdims=True)

    def __update_weights(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        """Updates weights and bias.

        At each iteration of gradient descent, the weights and bias are updated using the formula:
            w_{k+1} = w_k - γ * ∇w E(w_k)
            b_{k+1} = b_k - γ * ∇b E(b_k)

            where:
                - w_k, b_k are the current weight and bias at iteration k,
                - γ is the learning rate, determining the step size in the direction of the negative gradient,
                - ∇w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k,
                - ∇b E(b_k) is the gradient of the cost function E with respect to the bias w at iteration k.
        """
        self.weights = self.weights - self.learning_rate * self.__get_gradient_w(inputs, targets, model_confidence)
        self.bias = self.bias - self.learning_rate * self.__get_gradient_b(targets, model_confidence)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                z: Union[np.ndarray, None] = None) -> float:
        """Target function.

        Cross-Entropy Loss:
            E = - (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * ln(y_k (x_i)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the target value for data point i and class k,
                - y_k (x_i) is model output after activation for data point i and class k.

        Numerically stable formula:
            E = (1 / N) Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^(z_il - c)) - (z_ik - c)),

            where:
                - N is the size of the data set,
                - K is the number of classes,
                - t_{ik} is the target value for data point i and class k,
                - z_{il} is the model output before activation for data point i and class l,
                - z is the model output before activation (matrix z),
                - c is maximum values for each data point in matrix z.

        Binary loss:
            E = - (1 / N) Σ(i=0 to N-1) t_i * ln(y(x_i)) + (1 - t_i) * ln(1 - y(x_i))

        Numerically stable formula:
            E = (1 / N) Σ(i=0 to N-1) ln(1 + e^-z_i) + (1 - t_i) * z_i if z_i >= 0
                (1 / N) Σ(i=0 to N-1) ln(1 + e^z_i) - t_i * z_i if z_i < 0

        Args:
            inputs: The input data.
            targets: The target data.
            z: The model output before activation. If None, it will be computed.

        Returns:
            float: The value of the target function.
        """
        z = self.__get_model_output(inputs) if z is None else z

        if self.experiment_config.loss_type == LossType.sigmoid:
            z_flattened = z.flatten()
            log_values = np.log1p(np.exp(-np.abs(z_flattened)))

            positive_z = z_flattened >= 0
            negative_z = ~positive_z

            loss = np.zeros_like(z_flattened)
            loss[positive_z] = log_values[positive_z] + (1 - targets[positive_z]) * z_flattened[positive_z]
            loss[negative_z] = log_values[negative_z] - targets[negative_z] * z_flattened[negative_z]
        else:
            z = z - np.max(z, axis=0)
            loss = (targets.T * (np.log(np.exp(z).sum(axis=0)) - z)).sum(axis=0)

        return loss.mean()

    def compute_metrics(self, inputs: np.ndarray, targets: np.ndarray,
                        model_confidence: Union[np.ndarray, None] = None) -> float:
        """Metrics calculation."""
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        average_precision = average_precision_score(targets, model_confidence[-1, :])
        return average_precision

    def one_hot_encoding(self, targets: np.ndarray) -> np.ndarray:
        """Creates matrix of one-hot encoding vectors for input targets.

        One-hot encoding vector representation:
            t_i^(k) = 1 if k = t_i otherwise  0,

            where:
                - k in [0, self.output_vector_dimension - 1],
                - t_i - target class of i-sample.
        """
        samples_num = targets.shape[0]
        one_hot_encoding = np.zeros((samples_num, self.output_vector_dimension))
        one_hot_encoding[np.arange(samples_num), targets] = 1
        return one_hot_encoding

    def __gradient_descent_step(self, inputs: np.ndarray, targets: np.ndarray) -> (float, np.ndarray):
        """Gradient descent step.

        One step of gradient descent includes:
            1. calculating the confidence of the model,
            2. calculating the value of the target function,
            3. updating the weights.

        Returns:
            float: The value of the target function.
            np.ndarray: The model output after activation.
        """
        model_confidence = self.get_model_confidence(inputs)
        loss = self.__target_function_value(inputs, targets)
        self.__update_weights(inputs, targets, model_confidence)
        return loss, model_confidence

    def check_early_stopping(self, loss_valid_previous: float, loss_valid: float, start_stopping: int) -> int:
        """Checks early stopping: updates start_stopping number if needed."""
        if loss_valid_previous - loss_valid < self.experiment_config.early_stopping['min_delta']:
            start_stopping += 1
        else:
            start_stopping = 0
        return start_stopping

    def log_gradient_descent_iteration(self, iteration: int, loss_train: float, loss_valid: float,
                                       metric_train: float, metric_valid: float):
        """Logs all params for gradient descent iteration."""
        self.params_logger.log_param(iteration, SetType.train, LoggingParamType.loss, loss_train)
        self.params_logger.log_param(iteration, SetType.validation, LoggingParamType.loss, loss_valid)
        self.params_logger.log_param(iteration, SetType.train, LoggingParamType.metric, metric_train)
        self.params_logger.log_param(iteration, SetType.validation, LoggingParamType.metric, metric_valid)

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        """Trains the model via the gradient descent.

        This iterative process aims to find the weights that minimize the cost function E(w).
        """
        if self.experiment_config.loss_type == LossType.sigmoid:
            targets_train_loss = targets_train
            targets_valid_loss = targets_valid
        else:
            targets_train_loss = self.one_hot_encoding(targets_train)
            targets_valid_loss = self.one_hot_encoding(targets_valid)

        pbar = tqdm(np.arange(self.start_iteration, self.num_iterations))
        ap_valid_best = self.compute_metrics(inputs_valid, targets_valid)
        loss_valid_previous, start_stopping = 0, 0
        num_samples = inputs_train.shape[0]

        for iteration in pbar:
            loss_train = []

            model_confidence_train = np.zeros((self.output_vector_dimension, num_samples))
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                loss_train_part, model_confidence_train_part = self.__gradient_descent_step(inputs_train[batch_indices],
                                                                                  targets_train_loss[batch_indices])
                loss_train.append(loss_train_part * batch_indices.shape[0])
                model_confidence_train[:, batch_indices] = model_confidence_train_part
            loss_train = np.sum(loss_train) / inputs_train.shape[0]

            ap_train = self.compute_metrics(inputs_train, targets_train, model_confidence_train)

            loss_valid = self.__target_function_value(inputs_valid, targets_valid_loss)
            ap_valid = self.compute_metrics(inputs_valid, targets_valid)

            pbar.set_description(f'Train loss: {loss_train}\tTrain AP: {ap_train}\tValid AP: {ap_valid}')
            self.log_gradient_descent_iteration(iteration, loss_train, loss_valid, ap_train, ap_valid)

            if ap_valid > ap_valid_best:
                self.save('best_checkpoint.pickle')
                ap_valid_best = ap_valid

            if iteration % self.experiment_config.save_model_iter == 0:
                self.save(f'checkpoint_{iteration}.pickle')

            if self.experiment_config.early_stopping:
                start_stopping = self.check_early_stopping(loss_valid_previous, loss_valid, start_stopping)
                loss_valid_previous = loss_valid
                if start_stopping > self.experiment_config.early_stopping['patience']:
                    print(f"Early stopped")
                    break

        self.params_logger.plot_params(LoggingParamType.loss)
        self.params_logger.plot_params(LoggingParamType.metric)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Returns model prediction."""
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions

    def save(self, filename):
        """Saves trained model."""
        write_file((self.weights, self.bias), os.path.join(self.experiment_config.checkpoints_dir, filename))

    def load(self, filepath):
        """Loads trained model."""
        self._init_weights(*read_file(filepath))

    def prepare_model(self) -> int:
        """Prepares model: checkpoint loading (if needed) and start iteration set up."""
        start_iteration = 0
        if self.experiment_config.load_model:
            try:
                self.load(self.experiment_config.load_model_path)
                print(f"Model loaded successfully from {self.experiment_config.load_model_path}")
                start_iteration = self.experiment_config.load_model_epoch + 1
            except FileNotFoundError:
                print(f"Model file not found at {self.experiment_config.load_model_path}. Using init weights.")
            except Exception as e:
                print(f"An error occurred while loading the model: {str(e)}. Using init weight.")
        return start_iteration
