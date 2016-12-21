#include "NeuralNetworkTrainer.h"

double NeuralNetworkTrainer::Train(vector<vector<double> > &input,
		vector<vector<double> > &output) {
	for (int j = 0; j < _hidden_nodes; ++j) {
		_input_weights[j].resize(_input_nodes);
		for (int i = 0; i < _input_nodes; ++i)
			_input_weights[j][i] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int k = 0; k < _output_nodes; ++k) {
		_hidden_weights[k].resize(_hidden_nodes);
		for (int j = 0; j < _hidden_nodes; ++j)
			_hidden_weights[k][j] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int j = 0; j < _hidden_nodes; ++j)
		_hidden_bias[j] = rand() / (double) RAND_MAX - 0.5;

	for (int k = 0; k < _output_nodes; ++k)
		_output_bias[k] = rand() / (double) RAND_MAX - 0.5;

	int iterations = 0;
	double mean_square_error = 1e9;

	while (mean_square_error > kAcceptableMeanSquareError
			&& iterations++ <= kTrainingIterations) {
		vector<double> hiddenValues(_hidden_nodes), outputValues(_input_nodes),
				hiddenDelta(_hidden_nodes), outputDelta(_input_nodes);

		mean_square_error = 0;
		for (int testcase = 0; testcase < input.size(); ++testcase) {
			for (int j = 0; j < _hidden_nodes; ++j) {
				hiddenValues[j] = CalculateActivationFunction(
						GetNetInput(input[testcase], _input_weights[j], _hidden_bias[j]));
			}
			for (int k = 0; k < _output_nodes; ++k) {
				outputValues[k] = CalculateActivationFunction(
						GetNetInput(hiddenValues, _hidden_weights[k], _output_bias[k]));
				outputDelta[k] = (output[testcase][k] - outputValues[k])
						* CalculateActivationFunctionDerivative(
								GetNetInput(hiddenValues, _hidden_weights[k], _output_bias[k]));
				mean_square_error += (output[testcase][k] - outputValues[k])
						* (output[testcase][k] - outputValues[k]) / _output_nodes;
			}
			for (int j = 0; j < _hidden_nodes; ++j) {
				for (int k = 0; k < _output_nodes; ++k)
					hiddenDelta[j] += outputDelta[k] * _hidden_weights[k][j];
				hiddenDelta[j] *= CalculateActivationFunctionDerivative(
						hiddenValues[j]);
			}
			for (int k = 0; k < _output_nodes; ++k) {
				for (int j = 0; j < _hidden_nodes; ++j)
					_hidden_weights[k][j] += kLearningRate * outputDelta[k]
							* hiddenValues[j];
				_output_bias[k] += kLearningRate * outputDelta[k];
			}
			for (int j = 0; j < _hidden_nodes; ++j) {
				for (int i = 0; i < _input_nodes; ++i)
					_input_weights[j][i] += kLearningRate * hiddenDelta[j]
							* input[testcase][i];
				_hidden_bias[j] += kLearningRate * hiddenDelta[j];
			}
		}
	}
	return mean_square_error;
}
