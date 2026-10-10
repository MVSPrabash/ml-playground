#pragma once

#include <vector>
#include "Layer.hpp"

class Model {
public:
    using Layers = std::vector<Layer>;

    Model();

    void addLayer(const Layer& l);

    [[nodiscard]] const Layers& layers() const noexcept;
    [[nodiscard]] int nLayers() const noexcept;

    [[nodiscard]] std::vector<double> forward(const std::vector<double>& x);

    void train(const std::vector<double>& x, std::vector<double>& y);
    
private:
    Layers layers_;
};