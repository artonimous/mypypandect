# refactored from - realpython build a neural network with python cli inputs

from typing import no_type_check

import matplotlib as ml
import numpy as np
import scikit_learn as sk
from dataclasses import dataclass

@dataclass
class NeuralNetwork:
    weights: numpy.ndarray = np.array([np.random.randn(), np.random.randn()])  # random init
    bias: numpy.ndarray = np.random.randn()  # random init
    learning_rate: float = learning_rate  # provided on initiation


    @staticmethod
    def _calculate_bias(self, error_prediction, prediction_n_layer, layer_n_bias):
        return error_prediction * prediction_n_layer * layer_n_bias


    @staticmethod
    def _sigmoid(self, x):
        """
        Applies the Sigmoid Function to the input.
        It is used in this context as the 'activation function'
        for interaction between layers.

        Args:
            x [float]: 

        Returns:
            float: 0 => return <= 1
        """
        return 1/(1 + np.exp(-x))


    @staticmethod
    def _dsigmoid(x):
        """Derivative of the sigmoid function using the Chain Rule

        Args:
            x ([type]): [description]
        
        Returns:
            float: 0 => return <= 1
        
        """
        return sigmoid(x) * (1 - sigmoid(x))  

    @staticmethod
        """[summary]
        """
    def _mean_square_error(prediction, target):
        return np.square(prediction - target)

    @staticmethod
    def _dmean_square(prediction, target):
        return 2 * (prediction - target)

    @staticmethod
    def _derror_dweights(self, derror_dprediction, dprediction_dlayer_n, weights_dlayer_n):
        return derror_dprediction * dprediction_dlayer_n * weights_dlayer_n
    

    def _make_prediction(self, input_vector):
        """Calculates the output for a single node on the following layer.

        Args:
            input_vector (dtype matrix): outputs for previous layer
            weights (float): [description]
            bias (float): [description]

            Returns:
                float: input for the following layer
            """
            
        layer_n = np.dot(input_vector, self.weights) + self.bias
        layer_nplusone = self._sigmoid(layer_n)
        return layer_nplusone
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
    
    @staticmethod
    def dprediction:
        pass
    

    def _compute_gradients(self, input_vector, target):
        layer_n = np.dot(input_vector, self.weights) + self.bias
        layer_nplusone = self._sigmoid(layer_n)
        prediction = layer_nplusone

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayern = self._dsigmoid(layer_n)
        dlayer1_dbias = 1
        weights_dlayer_n = (0 * self.weights) + (1 * input_vector)
        derror_dbias = (derror_dprediction * dprediction_dlayer_n * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer_n * dweights_dlayer_n)
        return derror_dbias, derror_dweights
    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)


learning_rate = 0.1
nn = NeuralNetwork(learning_rate)
nn.predict(input_vector)
