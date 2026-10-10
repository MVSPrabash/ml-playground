#include "ActivationFunction.hpp"

#include <stdexcept>
#include <cmath>

ActivationFunction::ActivationFunction()
    : act_{Activation::Linear}
{}

ActivationFunction::ActivationFunction(Activation activation)
    : act_{activation}
{}


void ActivationFunction::setActivation(Activation activation) {
    act_ = activation;
}

double ActivationFunction::operator()(double z) const {
    switch (act_) {
    case Activation::Linear:
        return z;
    case Activation::Sigmoid:
        return sigmoid(z);
    case Activation::ReLu:
        return relu(z);
    case Activation::Tanh:
        return tanh(z);
    }

    throw std::logic_error("Invalid Activation Function");
}

std::vector<double> ActivationFunction::operator()(
    const std::vector<double>& input
) const {

    std::vector<double> output(input.size());

    std::transform(
        input.begin(),
        input.end(),
        output.begin(),
        *this
    );

    return output;
}

std::vector<std::vector<double>> ActivationFunction::operator() (
    const std::vector<std::vector<double>>& input
) const {

    int m = input.size();

    std::vector<std::vector<double>> output(m);

    std::transform(
        input.begin(),
        input.end(),
        output.begin(),
        *this
    );

    return output;
}

double ActivationFunction::derivative(double z) const {
    switch (act_) {
    case Activation::Linear:
        return 1;
    case Activation::Sigmoid:
        return sigmoidDerivative(z);
    case Activation::ReLu:
        return reluDerivative(z);
    case Activation::Tanh:
        return tanhDerivative(z);
    }

    throw std::logic_error("Invalid Activation Function");
}


double ActivationFunction::sigmoid(double z) const {
    if (z < 0.0) {
        double e = std::exp(z);
        return e / (1.0 + e);
    }

    return 1.0 / (1.0 + std::exp(-z));
}

double ActivationFunction::sigmoidDerivative(double z) const {
    double Y = sigmoid(z);
    return Y * (1 - Y);
}

double ActivationFunction::relu(double z) const {
    return std::max(z, 0.0);
}

double ActivationFunction::reluDerivative(double z) const {
    return z > 0.0 ? 1.0 : 0.0;
}

double ActivationFunction::tanh(double z) const {
    return std::tanh(z);
}

double ActivationFunction::tanhDerivative(double z) const {
    double Y = std::tanh(z);
    return 1.0 - Y * Y;
}
