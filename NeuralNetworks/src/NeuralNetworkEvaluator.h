/*
 * NeuralNetworkEvaluator.h
 *
 *  Created on: Dec 21, 2016
 *      Author: mohammad
 */

#ifndef NEURALNETWORKEVALUATOR_H_
#define NEURALNETWORKEVALUATOR_H_

#include "NeuralNetwork.h"

class NeuralNetworkEvaluator: NeuralNetwork {
public:
	NeuralNetworkEvaluator(const vector<vector<double> >& input_weights,
			const vector<vector<double>>& hidden_weights,
			const vector<double>& hidden_bias, const vector<double>& output_bias) :
			_input_nodes(input_weights.size()), _hidden_nodes(hidden_weights.size()), _output_nodes(
					output_bias.size()) {
	}
	vector<double> CalculateOutput();
};

#endif /* NEURALNETWORKEVALUATOR_H_ */
