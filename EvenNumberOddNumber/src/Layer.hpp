#pragma once

#include <vector>

#include "ActivationFunction.hpp"

class Layer {
public:
    using Weight  = double;
    using Weights = std::vector<std::vector<Weight>>;
    using Bias    = std::vector<double>;


    /// @brief Initialize a Layer, set an Activation Function
    /// @param n_in Number of incoming connections
    /// @param n_out Number of Neuron in the layer
    /// @param activation Non Linear Activation Function type
    /// @param random Generate initial parameter randomly
    Layer(
        int n_in,
        int n_out,
        ActivationFunction activation = Activation::Linear,
        bool random = true
    );


    /// @brief Change the Activation Function of the Layer
    void setActivationFunction(ActivationFunction activation);

    [[nodiscard]] ActivationFunction activationFunction() const noexcept;


    /// @brief Forward pass the incoming values through this layer
    /// @param prevA Activation values of previous layer
    /// @return Computed Activation values of this layer
    [[nodiscard]]
    std::vector<std::vector<double>>
    forward(const std::vector<std::vector<double>>& prevA) const;


    [[nodiscard]] const Weights& weights() const noexcept;
    [[nodiscard]] const Bias& bias() const noexcept;

    void updateWeights(const Weights& weights);
    void updateBias(const Bias& bias);

    /// @brief set weight of ith input of jth neuron
    /// @param i Input index of the neuron
    /// @param j Neuron index in the layer
    /// @param weight New weight
    void setWeight(int i, int j, double weight);


    /// @brief set bias value of jth neuron
    /// @param j Neuron index in the layer
    /// @param bias New bias
    void setBias(int j, double bias);

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
        int M, int N, int K,
        double alpha,
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B,
        double beta,
        std::vector<std::vector<double>>& C
    ) const noexcept;
};
