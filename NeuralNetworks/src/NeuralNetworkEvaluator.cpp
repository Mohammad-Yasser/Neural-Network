#include "NeuralNetworkEvaluator.h"

vector<double> NeuralNetworkEvaluator::CalculateOutput(
		const vector<double>& input) {
	vector<double> hidden_values(_hidden_nodes);
	CalculateLayerValues(input, _input_weights, _hidden_bias, hidden_values);
	vector<double> output_values(_output_nodes);
	CalculateLayerValues(hidden_values, _hidden_weights, _output_bias,
			output_values);
	return output_values;
}
