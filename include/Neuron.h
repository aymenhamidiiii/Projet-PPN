#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
private:
    std::vector<double> weights; // Poids
    double bias;                 // Biais
    double output;               // Sortie après activation
    double gradient;             // Gradient pour le backward pass

public:
    // Constructeur
    explicit Neuron(int numInputs);

    // Passage avant
    double forward(const std::vector<double>& inputs);

    // Mise à jour des poids et biais
    void updateWeights(const std::vector<double>& inputs, double learningRate);

    // Setter pour le gradient
    void setGradient(double grad);

    // Getters
    const std::vector<double>& getWeights() const;
    double getGradient() const;
};

#endif // NEURON_H
