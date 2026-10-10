#include "Layer.hpp"

#include <random>
#include <cmath>
#include <stdexcept>

Layer::Layer(
    int n_in,
    int n_out,
    ActivationFunction activation,
    bool random
) {
    if (n_in <= 0 || n_out <= 0) {
        throw std::invalid_argument("Invalid input or output dimensions");
    }

    weights_ = Weights(n_in, std::vector<double>(n_out));
    bias_    = Bias(n_out);

    activationFunction_ = activation;

    if (!random) return ;

    std::random_device rd;
    std::mt19937_64 gen(rd());

    double xavier = std::sqrt(6.0 / (n_in + n_out));

    std::uniform_real_distribution weightDist (-xavier, +xavier);

    for (int i = 0; i < n_in; i++) {
        for (int j = 0; j < n_out; j++)
            weights_[i][j] = weightDist(gen);
    }
}

void Layer::setActivationFunction(ActivationFunction activation) {
    activationFunction_ = activation;
}

ActivationFunction Layer::activationFunction() const noexcept {
    return activationFunction_;
}


std::vector<std::vector<double>>
Layer::forward (
    const std::vector<std::vector<double> >& prevA
) const {
    if (prevA.size() == 0) {
        throw std::invalid_argument("prevA.size is 0");
    }

    for (const auto& vec : prevA) {
        if (vec.size() != n_in()) throw std::invalid_argument("prevA dimensions mismatch");
    }

    std::size_t m = prevA.size();

    std::vector<std::vector<double>> Y (m, bias_);

    // prevA:    m x n_in
    // weights_: n_in x n_out
    // Y:        m x n_out, initialized with bias_

    gemm(
        m, n_out(), n_in(),
        1.0,
        prevA, weights_,
        1.0,
        Y
    );
    
    return activationFunction_(Y);
}


const Layer::Weights &Layer::weights() const noexcept {
    return weights_;
}

const Layer::Bias &Layer::bias() const noexcept {
    return bias_;
}


void Layer::updateWeights(const Weights &weights) {
    if (weights.size() != weights_.size()) {
        throw std::invalid_argument("Bad weight matrix dimensions, new weight matrix mismatch with old weights dimension");
    }

    for (const auto& vec : weights) {
        if (vec.size() != n_out()) throw std::invalid_argument("Bad weight matrix dimensions");
    }
    
    weights_ = weights;
}

void Layer::updateBias(const Bias &bias) {
    if (bias.size() != bias_.size()) {
        throw std::invalid_argument("Bad bias matrix dimensions");
    }

    bias_ = bias;
}

void Layer::setWeight(int i, int j, double weight) {
    if (j < 0 || j >= n_out() || i < 0 || i >= n_in()) {
        throw std::invalid_argument("Invalid weight index, i or j value");
    }

    weights_[i][j] = weight;
}

void Layer::setBias(int j, double bias) {
    if (j < 0 || j >= n_out()) {
        throw std::invalid_argument("Invalid bias index");
    }

    bias_[j] = bias;
}


std::size_t Layer::n_in() const noexcept {
    return weights_.size();
}

std::size_t Layer::n_out() const noexcept {
    return bias_.size();
}


/// @brief General Matrix-Matrix Multiplication
void Layer::gemm(
    int M, int N, int K,
    double alpha,
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    double beta,
    std::vector<std::vector<double>>& C
) const noexcept {
    // A: M x K
    // B: K x N
    // C: M x N
    //
    // C = alpha * A * B + beta * C

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;

            for (int k = 0; k < K; ++k) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}
