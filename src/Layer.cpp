#include <cstddef> // Pour size_t
#include <iostream> // Pour std::cout et std::endl
#include "Layer.h"

// Constructeur : initialise les neurones
Layer::Layer(int numNeurons, int numInputsPerNeuron) {
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron);
    }

    // Initialiser le vecteur outputs avec le nombre de neurones
    outputs.resize(numNeurons, 0.0);
    std::cout << "Layer created with " << numNeurons 
          << " neurons and outputs size initialized to: " 
          << outputs.size() << std::endl;
}

// Passage en avant
std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    outputs.clear();
    outputs.reserve(neurons.size()); // Réserve la mémoire pour éviter les redimensionnements dynamiques
    for (Neuron& neuron : neurons) {
        outputs.push_back(neuron.forward(inputs));
    }
    return outputs;
}

// Mise à jour des poids et biais des neurones
void Layer::updateWeights(const std::vector<double>& inputs, double learningRate) {
    for (Neuron& neuron : neurons) {
        neuron.updateWeights(inputs, learningRate);
    }
}

// Récupérer les sorties des neurones
const std::vector<double>& Layer::getOutputs() const {
    return outputs;
}

// Setter pour les gradients
void Layer::setGradients(const std::vector<double>& gradients) {
    for (size_t i = 0; i < neurons.size(); ++i) {
        neurons[i].setGradient(gradients[i]);
    }
}

int Layer::getNumInputs() const {
    return neurons.empty() ? 0 : neurons[0].getWeights().size();
}