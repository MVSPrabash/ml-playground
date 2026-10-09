#include "Neuron.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>

std::vector<double> parameterize(int x, int p) {
    std::vector<double> X;
    
    while (x > 0) {
        X.push_back(x % 10);
        x /= 10;
    }

    if (int(X.size()) > p) {
        std::string err_msg = std::format("Input's digits count should be atmost p = {}", p);
        throw std::invalid_argument (err_msg);
    }

    while (int(X.size()) != p) {
        X.push_back(0);
    }
    std::ranges::reverse(X);

    return X;
}

int main() {
    std::cout << std::setprecision(22);

    std::cout << "--- EXPERIMENT #1 ---\n";

    std::cout << "\n--- MODEL CONFIGURATION PANEL ---\n";
    int p;
    std::cout << "p: ";
    std::cin >> p;

    double learningRate;
    std::cout << "learningRate: ";
    std::cin >> learningRate;

    Neuron neuron(Neuron::Weights(p), 0);
    neuron.setLearningRate(learningRate);

    std::cout << "Model initialized\n"
        << "p = " << p << '\n'
        << "learningRate = " << learningRate << '\n'
    ;

    std::cout << "\n--- TRAINING DASHBOARD ---\n";

    int m;
    std::cout << "m: ";
    std::cin >> m;

    int maxEpoch;
    std::cout << "maxEpoch: ";
    std::cin >> maxEpoch;

    std::string trainingFilename;
    std::cout << "Training dataset filename: ";
    std::cin >> trainingFilename;

    std::cout << "Loading " << trainingFilename << "...\n";
    std::ifstream fin(trainingFilename);

    std::vector<int> x(m);
    std::vector<double> Y(m);
    for (int i = 0; i < m; i++) {
        fin >> x[i] >> Y[i];
    }
    std::cout << "[DONE] Loaded training dataset\n";

    // parameterization of x
    std::cout << "[START] Parametrizing the input" << std::endl;
    std::vector<std::vector<double> > X(m);
    for (int i = 0; i < m; i++) {
        X[i] = parameterize(x[i], p);
    }
    std::cout << "[DONE] Parametrization finish" << std::endl;
    
    std::cout << "[START] Training...\n";

    for (int epoch = 0; epoch < maxEpoch; epoch++) {
        Neuron::Weights prev_weights = neuron.weights();
        double prev_bias = neuron.bias();
        
        std::cout << std::endl;
        double totalLoss = 0;
        for (int i = 0; i < m; i++) {
            totalLoss += neuron.train(X[i], Y[i]);
        }

        std::cout << "[DONE] Epoch #" << epoch + 1 << '\n';
        std::cout << "parameters:\n";
        std::cout << "b: " << neuron.bias() << '\n';
        std::cout << "w: ";
        const Neuron::Weights& weights = neuron.weights();
        for (Neuron::Weight weight : weights) {
            std::cout << weight << ' ';
        }
        std::cout << '\n';
        std::cout << "Average Loss value: " << totalLoss / m << '\n';

        double delMax = 0;
        for (int i = 0; i < p; i++) {
            delMax = std::max(delMax, std::abs(prev_weights[i] - neuron.weights()[i]));
        }
        delMax = std::max(delMax, std::abs(prev_bias - neuron.bias()));

        std::cout << "delMax: " << delMax << '\n';
    }
    
    std::cout << "[DONE] Training Completed\n";

    std::ofstream fout ("param.data");
    if (!fout.is_open()) {
        std::cerr << "Couldn't open param.data file\n";
        return 1;
    }

    Neuron::Weights weights = neuron.weights();
    double bias = neuron.bias();
    
    fout << "bias: " << bias << std::endl;
    for (Neuron::Weight weight : weights) {
        fout << weight << '\n';
    }
    fout << std::flush;

    std::cout << "Parameters saved to param.data\n";

    
    std::cout << "\n--- TESTING DASHBOARD ---\n";
    std::cout << "[Keyboard interrupt to stop]\n";

    for (;;) {
        int x;
        std::cin >> x;

        std::vector<double> X = parameterize(x, p);
        double Y_hat = neuron.predict(X);

        std::cout << Y_hat << std::endl;
    }

}