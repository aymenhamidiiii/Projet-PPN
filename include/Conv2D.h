#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>

class Conv2D {
private:
    int numFilters;
    int filterSize;
    int stride;
    int inputHeight;
    int inputWidth;
    int inputChannels;
    std::string paddingType;
    std::vector<std::vector<std::vector<std::vector<double>>>> filters;
    std::vector<double> biases;

    void initializeWeights();

public:
    Conv2D(int numFilters, int filterSize, int stride, int inputHeight, int inputWidth, int inputChannels, const std::string& paddingType);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> applyPadding(const std::vector<std::vector<std::vector<double>>>& input, int padSize);
};

#endif // CONV2D_H
