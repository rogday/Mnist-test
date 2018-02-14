#include <armadillo>
#include <fstream>
#include <iomanip>
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
	Col<double> x;
	Col<double> y;
};

int main() {
	srand(time(nullptr));

	ifstream imgs, labels;

	imgs.open("MNIST/imgs/data", ios::binary);
	labels.open("MNIST/labels/data", ios::binary);

	labels.seekg(8);
	imgs.seekg(4);

	unsigned char r[4];
	int32_t num[3]{}; // amount, dim
	int8_t c;

	for (int i = 0; i < 3; ++i) {
		imgs.read((char *)r, 4);

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

	/*Dataset loaded*/

	NeuralNetwork nn({num[1] * num[2], 16, 16, 10}, 0.1);

	for (int i = 0; i < 10; ++i) {
		random_shuffle(dataset.begin(), dataset.end());
		for (auto &i : dataset)
			nn.train(i.x, i.y);
	}

	int cnt = 0;

	for (auto &i : dataset)
		if (i.y.index_max() == nn.guess(i.x).index_max())
			++cnt;

	cout << cnt << " from " << num[0] << " is " << (double)cnt / num[0] * 100
		 << '%' << endl;

	return 0;
}
