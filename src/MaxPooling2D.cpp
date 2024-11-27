#include "MaxPooling2D.h"
#include <limits>
#include <algorithm>
#include <string>
// Constructeur
MaxPooling2D::MaxPooling2D(int poolSize, int stride, const std::string& padding)
    : poolSize(poolSize), stride(stride), padding(padding) {}

// Calculer la hauteur de sortie
int MaxPooling2D::getOutputHeight(int inputHeight) const {
    if (padding == "same") {
        return (inputHeight + stride - 1) / stride;
    } else {
        return (inputHeight - poolSize + stride) / stride;
    }
}

// Calculer la largeur de sortie
int MaxPooling2D::getOutputWidth(int inputWidth) const {
    if (padding == "same") {
        return (inputWidth + stride - 1) / stride;
    } else {
        return (inputWidth - poolSize + stride) / stride;
    }
}

// Appliquer le pooling
std::vector<std::vector<std::vector<double>>> MaxPooling2D::forward(
    const std::vector<std::vector<std::vector<double>>>& input) {
    int inputChannels = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();

    int outputHeight = getOutputHeight(inputHeight);
    int outputWidth = getOutputWidth(inputWidth);

    std::vector<std::vector<std::vector<double>>> output(inputChannels,
        std::vector<std::vector<double>>(outputHeight, std::vector<double>(outputWidth, 0.0)));

    for (int c = 0; c < inputChannels; ++c) {
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                double maxVal = -std::numeric_limits<double>::infinity();
                for (int py = 0; py < poolSize; ++py) {
                    for (int px = 0; px < poolSize; ++px) {
                        int inY = y * stride + py;
                        int inX = x * stride + px;
                        if (inY < inputHeight && inX < inputWidth) {
                            maxVal = std::max(maxVal, input[c][inY][inX]);
                        }
                    }
                }
                output[c][y][x] = maxVal;
            }
        }
    }
    return output;
}
