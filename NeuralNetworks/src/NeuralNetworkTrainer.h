/*
 * NeuralNetworkTrainer.h
 *
 *  Created on: Dec 21, 2016
 *      Author: mohammad
 */

#ifndef NEURALNETWORKTRAINER_H_
#define NEURALNETWORKTRAINER_H_

#include "NeuralNetwork.h"

class NeuralNetworkTrainer: public NeuralNetwork {
public:
	void Train(vector<vector<double> > &input, vector<vector<double> > &output);
private:
	const int kTrainingIterations = 500;
	const double kAcceptableMeanSquareError = 100;
};

#endif /* NEURALNETWORKTRAINER_H_ */
