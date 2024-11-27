#ifndef MLP_H
#define MLP_H

#include <vector>
#include "Layer.h"

class MLP {
public:
    // Constructeur qui prend une liste de tailles de couches
    MLP(const std::vector<int>& layer_sizes);

    // Propagation avant (forward pass)
    std::vector<double> forward(const std::vector<double>& input);

private:
    std::vector<Layer> layers; // Les couches du réseau
};

#endif
