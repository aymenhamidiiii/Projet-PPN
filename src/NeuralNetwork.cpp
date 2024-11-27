#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>

// Constructeur
NeuralNetwork::NeuralNetwork(double learningRate) : learningRate(learningRate) {}

// Ajouter une couche au réseau
void NeuralNetwork::addLayer(int numNeurons, int numInputsPerNeuron) {
    if (layers.empty()) {
        // Ajouter la première couche avec le nombre d'entrées spécifié
        layers.emplace_back(numNeurons, numInputsPerNeuron);
    } else {
        // Ajouter les couches suivantes avec le nombre de sorties de la couche précédente
        int previousLayerSize = layers.back().getOutputs().size();
        layers.emplace_back(numNeurons, previousLayerSize);
    }

    // Vérification et message après l'ajout de la couche
    const Layer& addedLayer = layers.back();
    std::cout << "Added layer: " << numNeurons << " neurons, " 
              << numInputsPerNeuron << " inputs per neuron." << std::endl;
    std::cout << "Layer outputs initialized with size: " 
              << addedLayer.getOutputs().size() << std::endl;
}




// Passage en avant
std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs) {
    std::vector<double> currentOutputs = inputs;
    for (Layer& layer : layers) {
        currentOutputs = layer.forward(currentOutputs);
    }
    return currentOutputs;
}

// Calcul de la fonction de perte (entropie croisée pour classification)
double NeuralNetwork::computeLoss(const std::vector<double>& predicted, const std::vector<double>& actual) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss += actual[i] * std::log(predicted[i]) + (1 - actual[i]) * std::log(1 - predicted[i]);
    }
    return -loss / predicted.size();
}
// Rétropropagation des erreurs
void NeuralNetwork::backward(const std::vector<double>& expected) {
    if (expected.size() != layers.back().getOutputs().size()) {
        std::cerr << "Error: Size of expected outputs (" << expected.size()
                  << ") does not match size of output layer (" 
                  << layers.back().getOutputs().size() << ")" << std::endl;
        return;
    }

    std::vector<double> errors = expected;

    for (int i = layers.size() - 1; i >= 0; --i) {
        Layer& layer = layers[i];
        std::vector<double> gradients;

        const std::vector<double>& outputs = layer.getOutputs();

        if (i == static_cast<int>(layers.size()) - 1) { // Couche de sortie
            for (size_t j = 0; j < outputs.size(); ++j) {
                double error = outputs[j] - expected[j];
                gradients.push_back(error * outputs[j] * (1 - outputs[j])); // Gradient de la couche de sortie
            }
        } else { // Couches cachées
            const std::vector<Neuron>& nextLayerNeurons = layers[i + 1].getNeurons();
            std::vector<double> previousErrors(layer.getOutputs().size(), 0.0);

            for (size_t j = 0; j < outputs.size(); ++j) {
                double error = 0.0;
                for (size_t k = 0; k < nextLayerNeurons.size(); ++k) {
                    error += nextLayerNeurons[k].getGradient() * nextLayerNeurons[k].getWeights()[j];
                }
                gradients.push_back(error * outputs[j] * (1 - outputs[j])); // Gradient de la couche cachée
            }
            errors = previousErrors; // Mettre à jour les erreurs pour la couche précédente
        }

        // Appliquer les gradients à la couche actuelle
        layer.setGradients(gradients);
    }

    // Mise à jour des poids et des biais
    std::vector<double> inputs;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (i == 0) {
            inputs = layers[i].getOutputs(); // Données d'entrée
        } else {
            inputs = layers[i - 1].getOutputs();
        }
        layers[i].updateWeights(inputs, learningRate);
    }
}



// Entraîner le réseau
void NeuralNetwork::train(const std::vector<std::vector<double>>& trainInputs,
                          const std::vector<std::vector<double>>& trainOutputs,
                          int epochs) {
    if (trainInputs.empty() || trainOutputs.empty()) {
        std::cerr << "Error: Training data is empty!" << std::endl;
        return;
    }

    // Vérification de la taille d'entrée pour la première couche
    int expectedInputSize = layers[0].getNumInputs();
    if (trainInputs[0].size() != static_cast<size_t>(expectedInputSize)) {
        std::cerr << "Error: trainInput[0] size (" << trainInputs[0].size() 
                  << ") does not match input layer size (" 
                  << expectedInputSize << ")" << std::endl;
        return;
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < trainInputs.size(); ++i) {
            std::vector<double> predicted = forward(trainInputs[i]);
            totalLoss += computeLoss(predicted, trainOutputs[i]);
            backward(trainOutputs[i]);
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / trainInputs.size() << std::endl;
    }
}




// Prédiction
std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) {
    return forward(inputs);
}
