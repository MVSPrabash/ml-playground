#pragma once

#include <vector>

#include "Layer.hpp"
#include "Types.hpp"

class Model {
public:
    using Layers = std::vector<Layer>;

    Model();

    void addLayer(const Layer& l);

    [[nodiscard]] const Layers& layers() const noexcept;
    [[nodiscard]] std::size_t nLayers() const noexcept;

    [[nodiscard]] Math::Matrix forward(const Math::Matrix& x);

    void train(const Math::Matrix& x, const Math::Matrix& y);
    
private:
    Layers layers_;
};