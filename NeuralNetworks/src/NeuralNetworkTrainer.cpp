#include "NeuralNetworkTrainer.h"

double NeuralNetworkTrainer::CalculateActivationFunctionDerivative(double x) {
	return x * (1 - x);
}

void NeuralNetworkTrainer::InitializeWeights(int layer_size,
		int next_layer_size, vector<vector<double>>& weights) {
	weights.resize(next_layer_size);
	for (int j = 0; j < next_layer_size; ++j) {
		weights[j].resize(layer_size);
		for (int i = 0; i < layer_size; ++i) {
			weights[j][i] = rand() / (double) RAND_MAX * kWeightRangeSize
					- kWeightRangeSize / 2;
		}
	}
}

void NeuralNetworkTrainer::InitializeWeights() {
	InitializeWeights(_input_nodes, _hidden_nodes, _input_weights);
	InitializeWeights(_hidden_nodes, _output_nodes, _hidden_weights);
}

void NeuralNetworkTrainer::InitializeBiases() {
	_hidden_bias.resize(_hidden_nodes);
	for (int j = 0; j < _hidden_nodes; ++j)
		_hidden_bias[j] = rand() / (double) RAND_MAX * kBiasRangeSize
				- kBiasRangeSize / 2;

	_output_bias.resize(_output_nodes);
	for (int k = 0; k < _output_nodes; ++k)
		_output_bias[k] = rand() / (double) RAND_MAX * kBiasRangeSize
				- kBiasRangeSize / 2;
}

void NeuralNetworkTrainer::InitializeWeightsAndBiases() {
	InitializeWeights();
	InitializeBiases();
}

void NeuralNetworkTrainer::UpdateWeightsAndBiases(const vector<double>& delta,
		const vector<double>& values, vector<vector<double>>& weights,
		vector<double>& biases) {
	for (int j = 0; j < delta.size(); ++j) {
		for (int i = 0; i < values.size(); ++i) {
			weights[j][i] += kLearningRate * delta[j] * values[i];
		}
		biases[j] += kLearningRate * delta[j];
	}
}

double NeuralNetworkTrainer::Train(const vector<double>& input_values,
		const vector<double>& given_output_values) {
	vector<double> hidden_values(_hidden_nodes), output_values(_output_nodes),
			hidden_delta(_hidden_nodes), output_delta(_output_nodes);

	CalculateLayerValues(input_values, _input_weights, _hidden_bias,
			hidden_values);
	CalculateLayerValues(hidden_values, _hidden_weights, _output_bias,
			output_values, false);

	for (int k = 0; k < _output_nodes; ++k) {
		output_delta[k] = (given_output_values[k] - output_values[k]);
		output_delta[k] *= CalculateActivationFunctionDerivative(CalculateActivationFunction(output_values[k]));
	}
	for (int j = 0; j < _hidden_nodes; ++j) {
		for (int k = 0; k < _output_nodes; ++k) {
			hidden_delta[j] += output_delta[k] * _hidden_weights[k][j];
		}
		hidden_delta[j] *= CalculateActivationFunctionDerivative(hidden_values[j]);
	}


	UpdateWeightsAndBiases(hidden_delta, input_values, _input_weights,
			_hidden_bias);
	UpdateWeightsAndBiases(output_delta, hidden_values, _hidden_weights,
			_output_bias);

	return CalculateMeanSquareError(output_values, given_output_values);
}

double NeuralNetworkTrainer::Train(const vector<vector<double> > &input,
		const vector<vector<double> > &output) {

	InitializeWeightsAndBiases();

	int iterations = 0;
	double mean_square_error = kAcceptableMeanSquareError + 1;

	while (mean_square_error > kAcceptableMeanSquareError
			&& ++iterations <= kTrainingIterations) {

		mean_square_error = 0;

		for (int test_case = 0; test_case < input.size(); ++test_case) {
			mean_square_error += Train(input[test_case], output[test_case])
					/ input.size();
		}
	}
	return mean_square_error;
}
