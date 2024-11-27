#include "Conv2D.h"
#include <cmath>
#include <random>

Conv2D::Conv2D(int numFilters, int filterSize, int stride, int inputHeight, int inputWidth, int inputChannels, const std::string& paddingType)
    : numFilters(numFilters), filterSize(filterSize), stride(stride),
      inputHeight(inputHeight), inputWidth(inputWidth),
      inputChannels(inputChannels), paddingType(paddingType) {
    initializeWeights();
}

void Conv2D::initializeWeights() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);

    filters.resize(numFilters);
    for (int i = 0; i < numFilters; ++i) {
        filters[i].resize(inputChannels);
        for (int j = 0; j < inputChannels; ++j) {
            filters[i][j].resize(filterSize, std::vector<double>(filterSize));
            for (int k = 0; k < filterSize; ++k) {
                for (int l = 0; l < filterSize; ++l) {
                    filters[i][j][k][l] = distribution(generator);
                }
            }
        }
    }

    biases.resize(numFilters, 0.0);
}

std::vector<std::vector<std::vector<double>>> Conv2D::applyPadding(const std::vector<std::vector<std::vector<double>>>& input, int padSize) {
    int paddedHeight = inputHeight + 2 * padSize;
    int paddedWidth = inputWidth + 2 * padSize;

    std::vector<std::vector<std::vector<double>>> paddedInput(
        inputChannels, std::vector<std::vector<double>>(paddedHeight, std::vector<double>(paddedWidth, 0.0)));

    for (int c = 0; c < inputChannels; ++c) {
        for (int y = 0; y < inputHeight; ++y) {
            for (int x = 0; x < inputWidth; ++x) {
                paddedInput[c][y + padSize][x + padSize] = input[c][y][x];
            }
        }
    }

    return paddedInput;
}

std::vector<std::vector<std::vector<double>>> Conv2D::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    int padSize = (paddingType == "valid") ? 0 : (filterSize - 1) / 2;
    auto paddedInput = applyPadding(input, padSize);

    int outputHeight = (inputHeight - filterSize + 2 * padSize) / stride + 1;
    int outputWidth = (inputWidth - filterSize + 2 * padSize) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(numFilters, std::vector<std::vector<double>>(outputHeight, std::vector<double>(outputWidth, 0.0)));

    for (int f = 0; f < numFilters; ++f) {
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                double sum = 0.0;
                for (int c = 0; c < inputChannels; ++c) {
                    for (int i = 0; i < filterSize; ++i) {
                        for (int j = 0; j < filterSize; ++j) {
                            int inY = y * stride + i;
                            int inX = x * stride + j;
                            sum += paddedInput[c][inY][inX] * filters[f][c][i][j];
                        }
                    }
                }
                output[f][y][x] = sum + biases[f];
            }
        }
    }

    return output;
}
