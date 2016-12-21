#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <bits/stdc++.h>
using namespace std;

class NeuralNetwork {
public:
	double CalculateActivationFunction(double x);
	double CalculateActivationFunctionDerivative(double x);
	double GetNetInput(vector<double> &input, vector<double> &weights,
			double bias);

	void SetInputNodes(int input_nodes) {
		_input_nodes = input_nodes;
	}
	void SetHiddenNodes(int hidden_nodes) {
		_hidden_nodes = hidden_nodes;
	}
	void SetOutputNodes(int output_nodes) {
		_output_nodes = output_nodes;
	}

protected:
	int _input_nodes, _hidden_nodes, _output_nodes;
	vector<vector<double> > _input_weights, _hidden_weights;
	vector<double> _hidden_bias, _output_bias;
};

#endif /* NEURALNETWORK_H_ */
