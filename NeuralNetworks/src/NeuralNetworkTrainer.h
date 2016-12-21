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
	// Returns the MSE.
	double Train(vector<vector<double> > &input, vector<vector<double> > &output);
	void GetWeightsAndBiases(vector<vector<double>>& input_weights,
			vector<vector<double>>& output_weights);
private:
	const int kTrainingIterations = 500;
	const double kAcceptableMeanSquareError = 100;
	const double kLearningRate = 0.001;
};

#endif /* NEURALNETWORKTRAINER_H_ */
