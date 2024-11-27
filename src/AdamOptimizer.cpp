#include "AdamOptimizer.h"

void AdamOptimizer::initialize(int outputSize, int inputSize) {
    m.resize(outputSize, std::vector<std::vector<double>>(inputSize, std::vector<double>(1, 0.0)));
    v.resize(outputSize, std::vector<std::vector<double>>(inputSize, std::vector<double>(1, 0.0)));
    timeStep = 0;
}

void AdamOptimizer::updateWeights(std::vector<std::vector<double>>& weights,
                                   const std::vector<std::vector<double>>& gradients) {
    timeStep++;

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            m[i][j][0] = beta1 * m[i][j][0] + (1.0 - beta1) * gradients[i][j];
            v[i][j][0] = beta2 * v[i][j][0] + (1.0 - beta2) * gradients[i][j] * gradients[i][j];

            double mHat = m[i][j][0] / (1.0 - std::pow(beta1, timeStep));
            double vHat = v[i][j][0] / (1.0 - std::pow(beta2, timeStep));

            weights[i][j] -= learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }
}
