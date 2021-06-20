# refactored from - realpython build a neural network with python cli inputs

from typing import no_type_check

import matplotlib as ml
import numpy as np
import scikit_learn as sk
from dataclasses import dataclass

@dataclass
class NeuralNetwork:
    weights: numpy.ndarray = np.array([np.random.randn(), np.random.randn()])  # random init
    bias: numpy.ndarray = np.random.randn(): float  # random init
    learning_rate = learning_rate: float


    @staticmethod
    def calculate_bias(self, error_prediction, prediction_n_layer, layer_n_bias):
        return error_prediction * prediction_n_layer * layer_n_bias


    @staticmethod
    def _sigmoid(self, x):
        """
        Aplies the Sigmoid Function to the input.
        It is used in this context as the 'activation function'
        for interaction between layers.

        Args:
            x [float]: 

        Returns:
            float: 0 => return <= 1
        """
        return 1/(1 + np.exp(-x))


    @staticmethod
    def dsigmoid(x):
        """Derivative of the sigmoid function using the Chain Rule

        Args:
            x ([type]): [description]
        
        Returns:
            float: 0 => return <= 1
        
        """
        return sigmoid(x) * (1 - sigmoid(x))  


    def mean_square_error(prediction, target):
        return np.square(prediction - target)


    def dmean_square(prediction, target):
        return 2 * (prediction - target)


    def derror_dweights(derror_dprediction, dprediction_dlayer1, dlayer1_weights):
        return derror_dprediction * dprediction_dlayer1 * dlayer1_weights
    

    def make_prediction(self, input_vector, self.weights, self.bias):
        """Calculates the output for a single node on the following layer.

        Args:
            input_vector (dtype matrix): outputs for previous layer
            weights (float): [description]
            bias (float): [description]

            Returns:
                float: input for the following layer
            """
            
        layer_n = np.dot(input_vector, self.weights) + self.bias
        layer_nplusone = sigmoid(layer_n)
        return layer_nplusone

    def _compute_gradients(self, input_vector, target):
        layer_n = np.dot(input_vector, self.weights) + self.bias
        layer_nplusone = self._sigmoid(layer_n)
        prediction = layer_nplusone

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )