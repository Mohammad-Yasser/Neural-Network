#include "NeuralNetwork.h"

double NeuralNetwork::CalculateActivationFunction(double x) {

}

double NeuralNetwork::CalculateActivationFunctionDerivative(double x) {

}
double NeuralNetwork::GetNetInput(vector<double> &input,
		vector<double> &weights, double bias) {
	double netI = 0;
	for (int i = 0; i < input.size(); ++i)
		netI += input[i] * weights[i];
	return netI + bias;
}
