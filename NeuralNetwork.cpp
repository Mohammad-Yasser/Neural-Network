#include "NeuralNetwork.h"

const double kEpsilon = 1e-9;

double NeuralNetwork::CalculateActivationFunction(double net_input) {

}

double NeuralNetwork::CalculateActivationFunctionDerivative(double x) {
	return (CalculateActivationFunction(x + kEpsilon)
			- CalculateActivationFunction(x - kEpsilon)) / (2 * kEpsilon);
}
double NeuralNetwork::GetNetInput(vector<double> &input,
		vector<double> &weights, double bias) {
	double netI = 0;
	for (int i = 0; i < input.size(); ++i)
		netI += input[i] * weights[i];
	return netI + bias;
}
