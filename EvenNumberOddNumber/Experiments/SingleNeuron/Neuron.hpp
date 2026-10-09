#pragma once

#include <vector>
#include <cstddef>

class Neuron {
public:
    using Weight = double;
    using Weights = std::vector<double>;

    Neuron();

    explicit Neuron(int p);

    Neuron(const Weights& weights, double bias);

    
    [[nodiscard]] std::size_t num_predictors() const noexcept;

    [[nodiscard]] const Weights& weights() const noexcept;

    [[nodiscard]] double bias() const noexcept;

    void setLearningRate(double learningRate);
    
    double train(const std::vector<double>& X, double Y);

    double predict(const std::vector<double>& X) const;

private:
    Weights weights_;
    double bias_{};

    double learningRate_{};

    [[nodiscard]] double sigmoid(double z) const noexcept;

    [[nodiscard]] double lossFunction(double Y, double Y_hat) const noexcept;

};