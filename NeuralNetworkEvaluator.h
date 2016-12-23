/*
 * NeuralNetworkEvaluator.h
 *
 *  Created on: Dec 21, 2016
 *      Author: mohammad
 */

#ifndef NEURALNETWORKEVALUATOR_H_
#define NEURALNETWORKEVALUATOR_H_

#include "NeuralNetwork.h"

class NeuralNetworkEvaluator: public NeuralNetwork {
public:
//	NeuralNetworkEvaluator() = delete;
	NeuralNetworkEvaluator(const NeuralNetwork& network) :
			NeuralNetwork(network) {
	}
	vector<double> CalculateOutput(const vector<double>& input);
};

#endif /* NEURALNETWORKEVALUATOR_H_ */
