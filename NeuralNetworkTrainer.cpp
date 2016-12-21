#include "NeuralNetworkTrainer.h"

void NeuralNetworkTrainer::Train(vector<vector<double> > &input,
		vector<vector<double> > &output) {
	for (int j = 0; j < _hidden_nodes; ++j) {
		inputWeights[j].resize(_input_nodes);
		for (int i = 0; i < _input_nodes; ++i)
			inputWeights[j][i] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int k = 0; k < _output_nodes; ++k) {
		hiddenWeights[k].resize(_hidden_nodes);
		for (int j = 0; j < _hidden_nodes; ++j)
			hiddenWeights[k][j] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int j = 0; j < _hidden_nodes; ++j)
		hiddenBias[j] = rand() / (double) RAND_MAX - 0.5;

	for (int k = 0; k < _output_nodes; ++k)
		outputBias[k] = rand() / (double) RAND_MAX - 0.5;

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
						GetNetInput(input[testcase], inputWeights[j], hiddenBias[j]));
			}
			for (int k = 0; k < _output_nodes; ++k) {
				outputValues[k] = CalculateActivationFunction(
						GetNetInput(hiddenValues, hiddenWeights[k], outputBias[k]));
				outputDelta[k] = (output[testcase][k] - outputValues[k])
						* CalculateActivationFunctionDerivative(
								GetNetInput(hiddenValues, hiddenWeights[k], outputBias[k]));
				mean_square_error += (output[testcase][k] - outputValues[k])
						* (output[testcase][k] - outputValues[k]) / _output_nodes;
			}
			for (int j = 0; j < _hidden_nodes; ++j) {
				for (int k = 0; k < _output_nodes; ++k)
					hiddenDelta[j] += outputDelta[k] * hiddenWeights[k][j];
				hiddenDelta[j] *= CalculateActivationFunctionDerivative(
						hiddenValues[j]);
			}
			for (int k = 0; k < _output_nodes; ++k) {
				for (int j = 0; j < _hidden_nodes; ++j)
					hiddenWeights[k][j] += learningRate * outputDelta[k]
							* hiddenValues[j];
				outputBias[k] += learningRate * outputDelta[k];
			}
			for (int j = 0; j < _hidden_nodes; ++j) {
				for (int i = 0; i < _input_nodes; ++i)
					inputWeights[j][i] += learningRate * hiddenDelta[j]
							* input[testcase][i];
				hiddenBias[j] += learningRate * hiddenDelta[j];
			}
		}
	}
}
