#ifndef MAXPOOLING2D_H
#define MAXPOOLING2D_H
#include <string>
#include <vector>

class MaxPooling2D {
private:
    int poolSize;    // Taille de la fenêtre de pooling (assume carrée : poolSize x poolSize)
    int stride;      // Pas de déplacement
    std::string padding; // "valid" ou "same"

public:
    MaxPooling2D(int poolSize, int stride = 2, const std::string& padding = "valid");

    // Appliquer le pooling
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input);

    // Calculer les dimensions de sortie
    int getOutputHeight(int inputHeight) const;
    int getOutputWidth(int inputWidth) const;
};

#endif // MAXPOOLING2D_H
