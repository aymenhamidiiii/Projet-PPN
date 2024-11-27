#include "LossFunctions.h"
#include <cmath>
#include <algorithm>
#include <iostream>

double computeCrossEntropyLoss(const std::vector<double>& predictions, const std::vector<double>& labels) {
    double loss = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
        double prob = std::max(predictions[i], 1e-8); // Éviter log(0)
        loss -= labels[i] * std::log(prob);
    }
    return loss / labels.size();
}

std::vector<double> computeSoftmaxGradient(const std::vector<double>& predictions, const std::vector<double>& labels) {
    std::vector<double> gradients(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        gradients[i] = predictions[i] - labels[i]; // Gradient : (y_pred - y_true)
    }
    return gradients;
}