#include "Model.hpp"

#include <stdexcept>

Model::Model (
    std::size_t inputSize,
    std::initializer_list<LayerConfig> layers
) : inputSize_{inputSize} {

    std::size_t prevLayerSize = inputSize;
    for (const LayerConfig& layerConf : layers) {
        layers_.emplace_back(
            prevLayerSize,
            layerConf.n_out,
            layerConf.activation,
            layerConf.random
        );
        
        prevLayerSize = layerConf.n_out;
    }

    outputSize_ = prevLayerSize;
}

const Model::Layers& Model::layers() const noexcept {
    return layers_;
}

std::size_t Model::nLayers() const noexcept {
    return layers_.size();
}

Math::Matrix Model::forward(const Math::Matrix &x) const {
    if (x.size() == 0) return Math::Matrix();

    for (const Math::Vector& xi : x) {
        if (xi.size() != inputSize_) {
            throw std::invalid_argument(
                "Input dimension mismatch with Model's input layer"
            );
        }
    }
    
    Math::Matrix prediction = x;
    for (const Layer& layer : layers_) {
        prediction = layer.forward(prediction);
    }

    return prediction;
}

void Model::train(const Math::Matrix& x, const Math::Matrix& y) {
    
}
