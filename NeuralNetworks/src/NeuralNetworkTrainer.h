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

	NeuralNetworkTrainer(int input_nodes, int hidden_nodes, int output_nodes) :
			NeuralNetwork(input_nodes, hidden_nodes, output_nodes) {
	}
	void InitializeWeights(int layer_size, int next_layer_size,
			vector<vector<double>>& weights);
	void InitializeWeights();
	void InitializeBiases();
	void InitializeWeightsAndBiases();

	// The activation function used is the sigmoid function.
	double CalculateActivationFunctionDerivative(double x);

	// Returns the MSE.
	// "input" size must be equal to "output" size.
	double Train(const vector<vector<double> >& input,
			const vector<vector<double> >& output);
	double Train(const vector<double>& input, const vector<double>& output);

	void UpdateWeightsAndBiases(const vector<double>& delta,
			const vector<double>& values, vector<vector<double>>& weights,
			vector<double>& biases);
	void GetWeightsAndBiases(vector<vector<double>>& input_weights,
			vector<vector<double>>& output_weights);
private:
	const int kTrainingIterations = 500;
	const double kAcceptableMeanSquareError = 100;
	const double kLearningRate = 0.001;
	const double kWeightRangeSize = 100;
	const double kBiasRangeSize = 1;
};

#endif /* NEURALNETWORKTRAINER_H_ */
