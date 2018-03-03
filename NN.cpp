#include <armadillo>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#define ARMA_NO_DEBUG

using namespace std;
using namespace arma;

struct Data {
	Col<double> x;
	Col<double> y;
};

class NeuralNetwork {
  private:
	vector<Col<double>> L;
	vector<Mat<double>> W;
	double lr;

	static double activation(double x) { return 1.0 / (1.0 + exp(-x)); }

  public:
	NeuralNetwork(vector<int> &&vec, double lr)
		: L(vec.size()), W(L.size() - 1), lr(lr) {
		arma_rng::set_seed_random();
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

	int train(vector<Data> &dataset, int batchSize, int iterations) {
		int cnt;

		for (int j = 0; j < iterations; ++j) {
			random_shuffle(dataset.begin(), dataset.end());
			for (int i = 0; i < dataset.size() - batchSize; i += batchSize) {
				Col<double> E(dataset[0].y.n_rows, fill::zeros);

				for (int k = 0; k < batchSize; ++k)
					E += (dataset[i + k].y - guess(dataset[i + k].x)) /
						 batchSize;

				for (int k = W.size() - 1; k >= 0; --k) {
					W[k] += lr * E % L[k + 1] % (1 - L[k + 1]) * L[k].t();
					E = W[k].t() * E;
				}
			}

			cnt = 0;
			for (auto &i : dataset)
				if (i.y.index_max() == guess(i.x).index_max())
					++cnt;

			cout << j << ' ' << 100 * cnt / dataset.size() << endl;
		}

		return 100 * cnt / dataset.size();
	}
};

int main(int argc, char *argv[]) {
	ifstream imgs, labels;

	imgs.open("MNIST/imgs/data", ios::binary);
	labels.open("MNIST/labels/data", ios::binary);

	labels.seekg(8);
	imgs.seekg(4);

	unsigned char r[4];
	int32_t num[3]{}; // amount, dim
	int8_t c;

	for (int i = 0; i < 3; ++i) {
		imgs.read(reinterpret_cast<char *>(r), 4);

		for (int k = 0; k < 2; ++k)
			swap(r[k], r[3 - k]);

		for (int k = 0; k < 4; ++k)
			num[i] |= r[k] << (8 * k);
	}

	vector<Data> dataset(num[0]);

	for (int i = 0; i < num[0]; ++i) {

		Col<double> x(num[1] * num[2]);
		Col<double> y(10, fill::zeros);

		for (int k = 0; k < num[1] * num[2]; ++k) {
			imgs.read(reinterpret_cast<char *>(&c), sizeof c);
			x(k) = (double)c / 255.0;
		}

		labels.read(reinterpret_cast<char *>(&c), sizeof c);
		y(c) = 1.0;

		dataset[i].x = x;
		dataset[i].y = y;
	}

	NeuralNetwork nn({num[1] * num[2], 40, 10}, 0.1);
	cout << nn.train(dataset, 1, 60) << endl;

	return 0;
}
