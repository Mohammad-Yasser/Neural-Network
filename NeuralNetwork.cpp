#include "NeuralNetwork.h"

const double eps = 1e-9 ;
double NeuralNetwork::activationFunction(double netI) {

}
double NeuralNetwork::dydx(double x) {
	return (activationFunction(x + eps) - activationFunction(x - eps)) / (2 * eps);
}
double NeuralNetwork::sum(vector<double> &input, vector<double> &weights) {
	double netI = 0;
	for (int i = 0; i < input.size(); ++i)
		netI += input[i] * weights[i];
	return netI;
}
double NeuralNetwork::evaluate(vector<double> &input, vector<double> &weights) {
	return activationFunction(sum(input, weights));
}

void NeuralNetwork::train(vector<vector<double> > &input, vector<vector<double> > &output) {
	for (int j = 0; j < l; ++j) {
		inputWeights[j].resize(m);
		for (int i = 0; i < m; ++i)
			inputWeights[j][i] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int k = 0; k < n; ++k) {
		hiddenWeights[k].resize(l);
		for (int j = 0; j < l; ++j)
			hiddenWeights[k][j] = rand() / (double) RAND_MAX - 0.5;
	}
	for (int j = 0; j < l; ++j)
		hiddenBias[j] = rand() / (double) RAND_MAX - 0.5;

	for (int k = 0; k < n; ++k)
		outputBias[k] = rand() / (double) RAND_MAX - 0.5;

	int error = 1, iterations = 0;
	vector<double> hiddenValues(l), outputValues(m), hiddenDelta(l), outputDelta(m);
	while (error && iterations++ <= 30000) {
		error = 0;
		for (int testcase = 0; testcase < input.size(); ++testcase) {
			for (int j = 0; j < l; ++j)
				hiddenValues[j] = activationFunction(
				  hiddenBias[j] + sum(input[testcase], inputWeights[j]));
			for (int k = 0; k < m; ++k) {
				outputValues[k] = activationFunction(
				  outputBias[k] + sum(hiddenValues, hiddenWeights[k]));
				outputDelta[k] = (output[testcase][k] - outputValues[k])
				  * dydx(outputBias[k] + sum(hiddenValues, hiddenWeights[k]));
				error += (output[testcase][k] - outputValues[k])
				  * (output[testcase][k] - outputValues[k]);
			}
			for (int j = 0; j < l; ++j) {
				for (int k = 0; k < n; ++k)
					hiddenDelta[j] += outputDelta[k] * hiddenWeights[k][j];
				hiddenDelta[j] *= dydx(hiddenValues[j]);
			}
			for (int k = 0; k < n; ++k) {
				for (int j = 0; j < l; ++j)
					hiddenWeights[k][j] += learningRate * outputDelta[k] * hiddenValues[j];
				outputBias[k] += learningRate * outputDelta[k];
			}
			for (int j = 0; j < l; ++j) {
				for (int i = 0; i < m; ++i)
					inputWeights[j][i] += learningRate * hiddenDelta[j] * input[testcase][i];
				hiddenBias[j] += learningRate * hiddenDelta[j];
			}
		}
	}
}
