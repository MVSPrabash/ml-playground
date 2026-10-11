#include "Layer.hpp"

#include <random>
#include <cmath>
#include <stdexcept>

Layer::Layer(
    std::size_t n_in,
    std::size_t n_out,
    ActivationFunction activation,
    bool random
) {

    if (n_in == 0 || n_out == 0) {
        throw std::invalid_argument("Layer input or output size can't be zero");
    }

    weights_ = Weights(n_in, Math::Vector(n_out));
    bias_    = Bias(n_out);

    activationFunction_ = activation;

    if (!random) return ;

    std::random_device rd;
    std::mt19937_64 gen(rd());

    double xavier = std::sqrt(6.0 / (n_in + n_out));

    std::uniform_real_distribution weightDist (-xavier, +xavier);

    for (std::size_t i = 0; i < n_in; i++) {
        for (std::size_t j = 0; j < n_out; j++)
            weights_[i][j] = weightDist(gen);
    }
}

void Layer::setActivationFunction(ActivationFunction activation) {
    activationFunction_ = activation;
}

ActivationFunction Layer::activationFunction() const noexcept {
    return activationFunction_;
}


Math::Matrix Layer::forward (const Math::Matrix& prevA) const {
    if (prevA.size() == 0) {
        throw std::invalid_argument("prevA.size is 0");
    }

    for (const auto& vec : prevA) {
        if (vec.size() != n_in()) throw std::invalid_argument("prevA dimensions mismatch");
    }

    std::size_t m = prevA.size();

    Math::Matrix Y (m, bias_);

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

void Layer::setWeight(std::size_t i, std::size_t j, double weight) {
    if (i >= n_in() || j >= n_out()) {
        throw std::invalid_argument("Invalid weight index, i or j value");
    }

    weights_[i][j] = weight;
}

void Layer::setBias(std::size_t j, double bias) {
    if (j >= n_out()) {
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
    std::size_t M, std::size_t N, std::size_t K,
    double alpha,
    const Math::Matrix& A,
    const Math::Matrix& B,
    double beta,
    Math::Matrix& C
) const noexcept {
    // A: M x K
    // B: K x N
    // C: M x N
    //
    // C = alpha * A * B + beta * C

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;

            for (std::size_t k = 0; k < K; ++k) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}
