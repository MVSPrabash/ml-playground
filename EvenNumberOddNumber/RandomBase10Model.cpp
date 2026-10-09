// Compile: g++ RandomBase10Model.cpp -std=c++23 -Wall -O2 -D_GLIBCXX_DEBUG -o build/RandomBase10Model
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>


struct Model {
    std::vector<double> w;
    double b;

    Model() {}
    Model(int d) { w.resize(d); }
};

double sigmoid(double z) {
    if (z < 0.0) {
        double e = std::exp(z);
        return e / (1.0 + e);
    }

    return 1.0 / (1.0 + std::exp(-z));
}

int main() {
    int d;

    std::cout << "--- Random base10 Model ---\n";

    std::cout << "Input parameters\n";
    std::cout << "d: ";
    std::cin >> d;

    if (d <= 0) {
        throw std::invalid_argument("d must be atleast 1");
    }

    std::cout << "\n--- Input (# of digits, d = " << d << ") ---\n";

    int x;

    std::cout << "x: ";
    std::cin >> x;

    x = std::abs(x);

    // Parametrizing x
    std::cout << "\n--- Paramterized Input X ---\n";
    
    std::vector<double> X;
    {
        int x_ = x;
        while (x_ > 0) {
            X.push_back(x_ % 10);
            x_ /= 10;
        }

        if (int(X.size()) > d) {
            std::string err_msg = std::format("Input's digits count should be atmost d = {}", d);
            throw std::invalid_argument (err_msg);
        }

        while (int(X.size()) != d) {
            X.push_back(0);
        }
        std::ranges::reverse(X);
    }

    std::cout << "X: ";
    for (int i = 0; i < d; i++) {
        std::cout << X[i] << " \n"[i == d - 1];
    }

    // Model
    Model model (d);

    std::random_device rd;
    std::mt19937_64 gen(rd());  // Mersenne Twister engine
    std::uniform_real_distribution<double> weight_dist(-0.5, 0.5);

    for (int i = 0; i < d; i++) {
        model.w[i] = weight_dist(gen);
    }

    std::uniform_real_distribution<double> bias_dist(-0.1, 0.1);
    model.b = bias_dist(gen); 

    std::cout << "\n--- Model ---\n";
    std::cout << "w: ";
    for (int i = 0; i < d; i++) {
        std::cout << model.w[i] << " \n"[i == d - 1];
    }

    std::cout << "b: " << model.b << '\n';

    // Predict
    std::cout << "\n--- Forward Pass ---\n";

    std::cout << "Computing preactivation...\n";
    std::cout << "Eq: z = w^T.X + b\n";

    double z = std::inner_product(model.w.begin(), model.w.end(), X.begin(), model.b);

    std::cout << "w^T.X + b = z = " << z << std::endl;

    std::cout << "Computing sigmoid activation...\n";
    double Y = sigmoid(z);
    std::cout << "Y: " << Y << "\n";

    std::cout << "\n--- OUTPUT ---\n";
    std::cout << "Probability of Even: " << Y * 100 << "%\n";
    std::cout << "Probability of Odd : " << (1 - Y) * 100 << "%\n";    
}