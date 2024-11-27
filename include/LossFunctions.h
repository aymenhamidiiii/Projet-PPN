#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include <vector>

// Fonction pour calculer la perte d'entropie croisée
double computeCrossEntropyLoss(const std::vector<double>& predictions, const std::vector<double>& labels);

// Fonction pour calculer le gradient de softmax
std::vector<double> computeSoftmaxGradient(const std::vector<double>& predictions, const std::vector<double>& labels);

#endif