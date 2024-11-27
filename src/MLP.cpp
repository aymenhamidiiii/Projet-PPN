#include "MLP.h"
#include <cstddef>

MLP::MLP(const std::vector<int>& layer_sizes) {
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        // Ajouter une couche entre chaque taille indiquée
        layers.emplace_back(layer_sizes[i-1], layer_sizes[i]);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output); // Propagation dans chaque couche
    }
    return output;
}
