#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"
#include <vector>

class Layer {
private:
    std::vector<Neuron> neurons; // Neurones de la couche
    std::vector<double> outputs; // Sorties des neurones après activation

public:
    // Constructeur
    Layer(int numNeurons, int numInputsPerNeuron);

    // Passage en avant
    std::vector<double> forward(const std::vector<double>& inputs);

    // Mise à jour des poids pour chaque neurone
    void updateWeights(const std::vector<double>& inputs, double learningRate);

    // Récupérer les sorties
    const std::vector<double>& getOutputs() const;

    // Setter pour les gradients
    void setGradients(const std::vector<double>& gradients);

    // Getter pour accéder aux neurones
    const std::vector<Neuron>& getNeurons() const { return neurons; }

    int getNumInputs() const;
};

#endif // LAYER_H
