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
	void InitializeWeights();
	vector<double> CalculateOutput();
};

#endif /* NEURALNETWORKEVALUATOR_H_ */
