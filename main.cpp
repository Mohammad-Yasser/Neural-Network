#include <bits/stdc++.h>
#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"
#include "NeuralNetworkEvaluator.h"

using namespace std;

const int kScale = 1000;

int main() {
	freopen("training_data.in", "r", stdin);
	int input_nodes, hidden_nodes, output_nodes;
	cin >> input_nodes >> hidden_nodes >> output_nodes;

	int training_examples;
	cin >> training_examples;

	vector<vector<double>> input(training_examples);
	vector<vector<double>> output(training_examples);

	for (int i = 0; i < training_examples; ++i) {
		input[i].resize(input_nodes);
		for (double& value : input[i]) {
			cin >> value;
			value /= kScale;
		}

		output[i].resize(output_nodes);
		for (double& value : output[i]) {
			cin >> value;
			value /= kScale;
		}
	}

	NeuralNetworkTrainer trainer(input_nodes, hidden_nodes, output_nodes);
	cout << trainer.Train(input, output) << endl;
	NeuralNetworkEvaluator evaluator(trainer);
	for (int i = 0; i < 10; ++i) {
		auto output_ = evaluator.CalculateOutput(input[i]);
		cout << output_[0] * kScale << endl;
	}

}
