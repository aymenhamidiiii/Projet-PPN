#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork {
private:
    std::vector<Layer> layers;       // Couches du réseau
    double learningRate;             // Taux d'apprentissage
    double computeLoss(const std::vector<double>& predicted, const std::vector<double>& actual);

public:
    // Constructeur
    NeuralNetwork(double learningRate);

    // Ajouter une couche au réseau
    void addLayer(int numNeurons, int numInputsPerNeuron);

    // Passage en avant
    std::vector<double> forward(const std::vector<double>& inputs);

    // Rétropropagation des erreurs
    void backward(const std::vector<double>& expected);

    // Entraînement du réseau
    void train(const std::vector<std::vector<double>>& trainInputs,
               const std::vector<std::vector<double>>& trainOutputs,
               int epochs);

    // Prédiction
    std::vector<double> predict(const std::vector<double>& inputs);
};

#endif // NEURALNETWORK_H
