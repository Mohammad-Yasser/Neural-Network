#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <bits/stdc++.h>
using namespace std;



class NeuralNetwork {
public:
	double activationFunction(double netI);
	double dydx(double x);
	double sum(vector<double> &input, vector<double> &weights);
	double evaluate(vector<double> &input, vector<double> &weights);
	void train(vector<vector<double> > &input, vector<vector<double> > &output);

private:
	int m, l, n;
	// m = input layer nodes, l = hidden layer nodes, n = output layer nodes.
	// iterators on input, hidden, output layers are i,j,k respectively.
	vector<vector<double> > inputWeights, hiddenWeights;
	vector<double> hiddenBias, outputBias;
	double learningRate = 0.001;
	bool isActivationFunctionDiffFn = 1;
};

#endif /* NEURALNETWORK_H_ */
