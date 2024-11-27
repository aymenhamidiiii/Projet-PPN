#include "Neuron.h"
#include <random>
#include <algorithm>

// Constructeur
Neuron::Neuron(int numInputs) : bias(0.0), gradient(0.0) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);

    weights.resize(numInputs);
    for (double& weight : weights) {
        weight = distribution(generator);
    }
}

// Passage avant
double Neuron::forward(const std::vector<double>& inputs) {
    double sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * inputs[i];
    }
    output = std::max(0.0, sum); // Utilisation de ReLU
    return output;
}

// Mise à jour des poids et biais
void Neuron::updateWeights(const std::vector<double>& inputs, double learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learningRate * gradient * inputs[i];
    }
    bias -= learningRate * gradient;
}

// Setter pour le gradient
void Neuron::setGradient(double grad) {
    gradient = grad;
}

// Getter pour les poids
const std::vector<double>& Neuron::getWeights() const {
    return weights;
}

// Getter pour le gradient
double Neuron::getGradient() const {
    return gradient;
}
