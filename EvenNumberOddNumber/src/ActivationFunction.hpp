#pragma once

#include "Types.hpp"

#include <algorithm>

enum class Activation {
    Linear,
    Sigmoid,
    ReLU,
    Tanh
};

/// @todo calculate in-place for batch processing (efficient)

class ActivationFunction {
public:
    /// @brief ActivationFunction defaults to Activation::Linear
    ActivationFunction();

    ActivationFunction(Activation activation);

    void setActivation(Activation activation);

    [[nodiscard]] double operator()(double z) const;


    /// @brief Process activation values element-wise
    [[nodiscard]]
    Math::Vector operator()(const Math::Vector& input) const;


    /// @brief Batch process Activation values element-wise
    /// @param input Batch of input. dimension: batch_size x n
    /// @return Element-wise activation values
    [[nodiscard]]
    Math::Matrix operator() (const Math::Matrix& input) const;


    /// @brief Derivative of the Activation Function at z
    [[nodiscard]] double derivative(double z) const;

private:
    Activation act_;

    [[nodiscard]] double sigmoid(double z) const;
    [[nodiscard]] double sigmoidDerivative(double z) const;

    [[nodiscard]] double relu(double z) const;
    [[nodiscard]] double reluDerivative(double z) const;
    
    [[nodiscard]] double tanh(double z) const;
    [[nodiscard]] double tanhDerivative(double z) const;

};
