#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

class Dense {
public:
    Dense(int inputSize, int outputSize, const std::string& activation);

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);

    // Backward pass
    std::vector<double> backward(const std::vector<double>& dLoss_dOutput,
                                 const std::vector<double>& prevLayerOutput,
                                 double learningRate);

private:
    int inputSize;                     // Taille d'entrée
    int outputSize;                    // Taille de sortie
    std::string activation;            // Type d'activation (relu, softmax, etc.)
    std::vector<std::vector<double>> weights;  // Matrice de poids
    std::vector<double> biases;        // Biais
    std::vector<double> inputs;        // Entrées sauvegardées pour le backward pass
    std::vector<double> outputs;       // Sorties sauvegardées pour le backward pass

    // Initialisation des poids
    void initializeWeights();

    // Fonctions d'activation (déclaration uniquement, implémentées dans Dense.cpp)
    double relu(double x) const;
    double reluDerivative(double x) const;
    double leakyRelu(double x) const;
    double leakyReluDerivative(double x) const;
    std::vector<double> softmax(const std::vector<double>& x) const;

    // Gradient clipping
    double clip(double value, double minVal, double maxVal) const {
        return std::max(minVal, std::min(value, maxVal));
    }
};

#endif // DENSE_H
