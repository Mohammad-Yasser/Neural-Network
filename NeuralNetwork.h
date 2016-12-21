#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <bits/stdc++.h>
using namespace std;

class NeuralNetwork {
public:
	double CalculateActivationFunction(double netI);
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
	// input_nodes = input layer nodes, l = hidden layer nodes, n = output layer nodes.
	// iterators on input, hidden, output layers are i,j,k respectively.
	vector<vector<double> > inputWeights, hiddenWeights;
	vector<double> hiddenBias, outputBias;
	double learningRate = 0.001;
};

#endif /* NEURALNETWORK_H_ */
