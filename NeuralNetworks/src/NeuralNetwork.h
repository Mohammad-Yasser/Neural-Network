#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <bits/stdc++.h>
using namespace std;

class NeuralNetwork {
public:
	NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) :
			_input_nodes(input_nodes), _hidden_nodes(hidden_nodes), _output_nodes(
					output_nodes) {
	}

	NeuralNetwork() {

	}

	// The activation function used is the sigmoid function.
	double CalculateActivationFunction(double x);
	double GetNetInput(const vector<double>& input, const vector<double>& weights,
			double bias);
	// "layer_values" size must be equal to the number of layer's nodes.
	void CalculateLayerValues(const vector<double>& input,
			const vector<vector<double>>& weights, const vector<double>& biases,
			vector<double>& layer_values, bool apply_activation_function = true);

protected:
	int _input_nodes, _hidden_nodes, _output_nodes;
	// weights[i][j] means the weight on the edge between j-th node of this layer
	// to i-th node of the next layer.
	vector<vector<double> > _input_weights, _hidden_weights;
	vector<double> _hidden_bias, _output_bias;
};

// "first" size must be equal to "second" size.
double CalculateMeanSquareError(const vector<double>& first,
		const vector<double>& second);

#endif /* NEURALNETWORK_H_ */
