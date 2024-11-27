#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include <vector>
#include <cmath>
#include <algorithm>

class AdamOptimizer {
private:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    std::vector<std::vector<std::vector<double>>> m; // Moyenne des gradients
    std::vector<std::vector<std::vector<double>>> v; // Variance des gradients
    int timeStep; // Pas de temps

public:
    AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), timeStep(0) {}

    void initialize(int outputSize, int inputSize);

    void updateWeights(std::vector<std::vector<double>>& weights,
                       const std::vector<std::vector<double>>& gradients);
};

#endif
