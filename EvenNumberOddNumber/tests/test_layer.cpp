
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Layer.hpp"

bool approx(double actual, double expected, double eps = 1e-9) {
    return std::abs(actual - expected) < eps;
}

void test_linear() {
    Layer layer(2, 2, Activation::Linear, false);

    layer.updateWeights({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    layer.updateBias({0.5, -1.0});

    const auto Y = layer.forward({
        {1.0, 2.0},
        {-2.0, 1.0}
    });

    assert(approx(Y[0][0], 7.5));
    assert(approx(Y[0][1], 9.0));
    assert(approx(Y[1][0], 1.5));
    assert(approx(Y[1][1], -1.0));
}

void test_relu() {
    Layer layer(1, 2, Activation::ReLu, false);

    layer.updateWeights({{-2.0, 3.0}});
    layer.updateBias({0.0, -1.0});

    const auto Y = layer.forward({{1.0}, {-1.0}});

    assert(approx(Y[0][0], 0.0));
    assert(approx(Y[0][1], 2.0));
    assert(approx(Y[1][0], 2.0));
    assert(approx(Y[1][1], 0.0));
}

void test_sigmoid() {
    Layer layer(1, 1, Activation::Sigmoid, false);

    layer.updateWeights({{1.0}});
    layer.updateBias({0.0});

    const auto Y = layer.forward({{0.0}, {2.0}, {-2.0}});

    assert(approx(Y[0][0], 0.5));
    assert(approx(Y[1][0], 1.0 / (1.0 + std::exp(-2.0))));
    assert(approx(Y[2][0], 1.0 / (1.0 + std::exp(2.0))));
}

void test_tanh() {
    Layer layer(1, 1, Activation::Tanh, false);

    layer.updateWeights({{1.0}});
    layer.updateBias({0.0});

    const auto Y = layer.forward({{0.0}, {1.0}, {-1.0}});

    assert(approx(Y[0][0], 0.0));
    assert(approx(Y[1][0], std::tanh(1.0)));
    assert(approx(Y[2][0], std::tanh(-1.0)));
}

void test_single_input() {
    Layer layer(2, 1, Activation::Linear, false);

    layer.updateWeights({
        {2.0},
        {3.0}
    });
    layer.updateBias({1.0});

    const auto Y = layer.forward({{4.0, 5.0}});

    assert(Y.size() == 1);
    assert(Y[0].size() == 1);
    assert(approx(Y[0][0], 24.0));
}

void test_invalid_dimensions() {
    Layer layer(2, 1, Activation::Linear, false);

    bool threw = false;

    try {
        (void)layer.forward({{1.0, 2.0}, {3.0}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_batch() {
    Layer layer(2, 1, Activation::Linear, false);

    bool threw = false;

    try {
        (void)layer.forward({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

int main() {
    test_linear();
    test_relu();
    test_sigmoid();
    test_tanh();
    test_single_input();
    test_invalid_dimensions();
    test_empty_batch();

    std::cout << "All feed-forward tests passed!\n";
}
