#pragma once

#include <vector>
#include <initializer_list>

#include "Layer.hpp"
#include "Types.hpp"

struct LayerConfig {
    std::size_t n_out;
    Activation activation;
    bool random = true;
};


class Model {
public:
    using Layers = std::vector<Layer>;

    Model(
        std::size_t inputSize,
        std::initializer_list<LayerConfig> layers
    );

    [[nodiscard]] const Layers& layers() const noexcept;
    [[nodiscard]] std::size_t nLayers() const noexcept;

    [[nodiscard]] Math::Matrix forward(const Math::Matrix& x) const;

    void train(const Math::Matrix& x, const Math::Matrix& y);
    
private:
    std::size_t inputSize_;
    std::size_t outputSize_;
    Layers layers_;
};