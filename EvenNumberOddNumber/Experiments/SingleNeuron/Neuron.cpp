#include "Neuron.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <random>

Neuron::Neuron() {}

Neuron::Neuron(const int p) {
    if (p <= 0) {
        throw std::invalid_argument("p must be greater than 0");
    }

    std::random_device rg;
    std::mt19937_64 gen(rg());
    
    std::uniform_real_distribution weight_dist(-1.0, 1.0);

    weights_.resize(p);
    for (Neuron::Weight &weight : weights_) {
        weight = weight_dist(gen);
    }

    std::uniform_real_distribution bias_dist(-0.1, 0.1);

    bias_ = bias_dist(gen);
}

Neuron::Neuron(const Weights& weights, double bias) : weights_{weights}, bias_{bias} {}


[[nodiscard]]
std::size_t Neuron::num_predictors() const noexcept {
    return weights_.size();
}

[[nodiscard]]
const Neuron::Weights& Neuron::weights() const noexcept {
    return weights_;
}

[[nodiscard]]
double Neuron::bias() const noexcept {
    return bias_;
}

void Neuron::setLearningRate(double learningRate) {
    if (learningRate <= 0) {
        throw std::invalid_argument("learning rate must be greater than 0");
    }

    learningRate_ = learningRate;
}


/// @brief Train the Neuron with given input X and supervised output Y using Gradient Descent
/// @param X input 
/// @param Y supervised output
/// @return Loss function value
double Neuron::train(const std::vector<double>& X, double Y) {
    if (X.size() != num_predictors()) {
        throw std::invalid_argument("Input size is not equal to number of predictors\n");
    }
    
    double Y_hat = predict(X);

    double L = lossFunction(Y, Y_hat);

    // Gradients
    std::vector<double> dLdw (num_predictors());
    double dLdb = 0;

    for (int i = 0; i < num_predictors(); i++) {
        dLdw[i] = (Y_hat - Y) * X[i];
    }

    dLdb = Y_hat - Y;

    // Update weights
    for (int i = 0; i < num_predictors(); i++) {
        weights_[i] = weights_[i] - learningRate_ * dLdw[i];
    }

    bias_ = bias_ - learningRate_ * dLdb;

    return L;
}


/// @brief Predict Y using sigmoid(w^T.X + b)
/// @param X input
/// @return Prediction for X
double Neuron::predict(const std::vector<double>& X) const {
    if (X.size() != num_predictors()) {
        throw std::invalid_argument("Size of Input doesn't match the number of predictors");
    }

    double z = std::inner_product(weights_.begin(), weights_.end(), X.begin(), bias_);
    return sigmoid(z);
}

[[nodiscard]]
double Neuron::sigmoid(double z) const noexcept {
    if (z < 0.0) {
        double e = std::exp(z);
        return e / (1.0 + e);
    }

    return 1.0 / (1.0 + std::exp(-z));
}

/// @brief Binary Cross Entropy Loss Function
/// @param Y Original Value
/// @param Y_hat Predicted Value
/// @return cost
[[nodiscard]]
double Neuron::lossFunction(double Y, double Y_hat) const noexcept {
    return - (Y * std::log(Y_hat) + (1 - Y) * std::log(1 - Y_hat)); 
};
