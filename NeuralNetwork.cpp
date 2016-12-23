#include "NeuralNetwork.h"

double NeuralNetwork::CalculateActivationFunction(double x) {
	double res = 1 / (1 + exp(-x));
	return res;
}

// "first" size must be equal to "second" size.
double CalculateMeanSquareError(const vector<double>& first,
		const vector<double>& second) {
	double mean_square_error = 0;
	for (int i = 0; i < first.size(); ++i) {
		mean_square_error += pow(first[i] - second[i], 2);
	}
	mean_square_error /= first.size();
	return mean_square_error;
}

double NeuralNetwork::GetNetInput(const vector<double> &input,
		const vector<double> &weights, double bias) {
	double net_input = 0;
	for (int i = 0; i < input.size(); ++i) {
		net_input += input[i] * weights[i];
	}

	return net_input + bias;
}

void NeuralNetwork::CalculateLayerValues(const vector<double>& input,
		const vector<vector<double> >& weights, const vector<double>& biases,
		vector<double>& layer_values, bool apply_activation_function) {
	for (int i = 0; i < layer_values.size(); ++i) {
		layer_values[i] = GetNetInput(input, weights[i], biases[i]);
		if (apply_activation_function) {
			layer_values[i] = CalculateActivationFunction(layer_values[i]);
		}
	}
}
