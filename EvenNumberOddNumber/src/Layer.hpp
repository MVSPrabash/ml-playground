#pragma once

#include "ActivationFunction.hpp"
#include "Types.hpp"

class Layer {
public:
    using Weight  = Math::Scalar;
    using Weights = Math::Matrix;
    using Bias    = Math::Vector;


    /// @brief Initialize a Layer, set an Activation Function
    /// @param n_in Number of incoming connections
    /// @param n_out Number of Neuron in the layer
    /// @param activation Non Linear Activation Function type
    /// @param random Generate initial parameter randomly
    Layer(
        std::size_t n_in,
        std::size_t n_out,
        ActivationFunction activation = Activation::Linear,
        bool random = true
    );


    /// @brief Change the Activation Function of the Layer
    void setActivationFunction(ActivationFunction activation);

    [[nodiscard]] ActivationFunction activationFunction() const noexcept;


    /// @brief Forward pass the incoming values through this layer
    /// @param prevA Activation values of previous layer
    /// @return Computed Activation values of this layer
    [[nodiscard]] Math::Matrix forward(const Math::Matrix& prevA) const;


    [[nodiscard]] const Weights& weights() const noexcept;
    [[nodiscard]] const Bias& bias() const noexcept;

    void updateWeights(const Weights& weights);
    void updateBias(const Bias& bias);

    /// @brief set weight of ith input of jth neuron
    /// @param i Input index of the neuron
    /// @param j Neuron index in the layer
    /// @param weight New weight
    void setWeight(std::size_t i, std::size_t j, double weight);


    /// @brief set bias value of jth neuron
    /// @param j Neuron index in the layer
    /// @param bias New bias
    void setBias(std::size_t j, double bias);

    /// @brief input dimension or previous layer size
    [[nodiscard]] std::size_t n_in() const noexcept;

    /// @brief output dimension or number of neurons in the Layer
    [[nodiscard]] std::size_t n_out() const noexcept;

private:
    Weights weights_;   // n_in X n_out
    Bias    bias_;      // n_out

    ActivationFunction activationFunction_;

    // inspired by CUDA
    void gemm(
        std::size_t M, std::size_t N, std::size_t K,
        double alpha,
        const Math::Matrix& A,
        const Math::Matrix& B,
        double beta,
        Math::Matrix& C
    ) const noexcept;
};
