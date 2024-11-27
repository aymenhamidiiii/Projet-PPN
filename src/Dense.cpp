#include "Dense.h"
#include <cmath>
#include <random>
#include <limits>

// Constructeur
Dense::Dense(int inputSize, int outputSize, const std::string& activation)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    initializeWeights();
    biases.resize(outputSize, 0.0);
}

// Initialisation des poids
void Dense::initializeWeights() {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    weights.resize(outputSize, std::vector<double>(inputSize));
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = distribution(generator);
        }
    }
}

// Forward pass
std::vector<double> Dense::forward(const std::vector<double>& input) {
    inputs = input;
    outputs.resize(outputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        outputs[i] = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            outputs[i] += weights[i][j] * input[j];
        }

        // Appliquer l'activation
        if (activation == "relu") {
            outputs[i] = relu(outputs[i]);
        } else if (activation == "leaky_relu") {
            outputs[i] = leakyRelu(outputs[i]);
        }
    }

    if (activation == "softmax") {
        outputs = softmax(outputs);
    }

    return outputs;
}

// Backward pass
std::vector<double> Dense::backward(const std::vector<double>& dLoss_dOutput,
                                    const std::vector<double>& prevLayerOutput,
                                    double learningRate) {
    std::vector<double> dLoss_dInput(inputSize, 0.0);

    for (int i = 0; i < outputSize; ++i) {
        double grad = dLoss_dOutput[i];
        if (activation == "relu") {
            grad *= reluDerivative(outputs[i]);
        } else if (activation == "leaky_relu") {
            grad *= leakyReluDerivative(outputs[i]);
        }

        for (int j = 0; j < inputSize; ++j) {
            dLoss_dInput[j] += grad * weights[i][j];
            weights[i][j] -= learningRate * grad * prevLayerOutput[j];
        }

        biases[i] -= learningRate * grad;
    }

    return dLoss_dInput;
}

// Fonctions d'activation
double Dense::relu(double x) const {
    return std::max(0.0, x);
}

double Dense::reluDerivative(double x) const {
    return (x > 0.0) ? 1.0 : 0.0;
}

double Dense::leakyRelu(double x) const {
    return (x > 0.0) ? x : 0.01 * x;
}

double Dense::leakyReluDerivative(double x) const {
    return (x > 0.0) ? 1.0 : 0.01;
}

std::vector<double> Dense::softmax(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    double maxVal = *std::max_element(x.begin(), x.end());
    double sumExp = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - maxVal);
        sumExp += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sumExp;
    }

    return result;
}
