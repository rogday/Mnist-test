#include <armadillo>
#include <iostream>
#include <vector>

using namespace std;
using namespace arma;

class NeuralNetwork {
  private:
	vector<Col<double>> L;
	vector<Mat<double>> W;
	double lr;

	static double activation(double x) { return 1.0 / (1.0 + exp(-x)); }

  public:
	NeuralNetwork(vector<int> &&vec, double lr)
		: L(vec.size()), W(L.size() - 1), lr(lr) {
		int n = vec.size() - 1;

		for (int i = 0; i < n; ++i) {
			L[i] = dcolvec(vec[i] + 1);
			L[i](vec[i]) = 1;
		}

		L[n] = colvec(vec[n]);

		for (int i = 0; i < n; ++i)
			W[i] = dmat(L[i + 1].n_rows, L[i].n_rows, fill::randu);
	}

	Col<double> guess(Col<double> &input) {
		for (int i = 0; i < input.size(); ++i)
			L[0](i) = input(i);

		for (int i = 1; i < L.size(); ++i) {
			L[i] = (W[i - 1] * L[i - 1]);
			L[i].for_each([](double &x) { x = activation(x); });
		}

		return L[L.size() - 1];
	}

	void train(Col<double> &input, Col<double> &answer) {
		Col<double> E = answer - guess(input);

		for (int i = W.size() - 1; i >= 0; --i) {
			W[i] += lr * E % L[i + 1] % (1 - L[i + 1]) * L[i].t();
			E = W[i].t() * E;
		}
	}
};

struct Data {
	Col<double> input;
	Col<double> answer;
};

int main() {
	srand(time(nullptr));

	NeuralNetwork nn({2, 208, 1}, 0.1);

	vector<Data> dataset = {
		{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};

	for (int i = 0; i < 5000; ++i) {
		random_shuffle(dataset.begin(), dataset.end());
		for (auto &i : dataset)
			nn.train(i.input, i.answer);
	}

	sort(dataset.begin(), dataset.end(), [](Data &a, Data &b) {
		if (a.input(0) != b.input(0))
			return a.input(0) < b.input(0);
		else
			return a.input(1) < b.input(1);
	});

	for (auto &i : dataset)
		cout << nn.guess(i.input) << endl;

	return 0;
}
