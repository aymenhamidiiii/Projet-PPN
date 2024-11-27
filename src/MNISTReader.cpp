// src/MNISTReader.cpp
#include "MNISTReader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

void readMNISTImages(const std::string &filename, std::vector<std::vector<double>> &images, int &numImages, int &imageSize) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    uint32_t magicNumber;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber);
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    numImages = __builtin_bswap32(numImages);

    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = __builtin_bswap32(rows);
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = __builtin_bswap32(cols);

    imageSize = rows * cols;

    // Redimensionner le vecteur d'images
    images.resize(numImages, std::vector<double>(imageSize));

    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0; // Normalisation entre 0 et 1
        }
    }

    file.close();
}


void readMNISTLabels(const std::string &filename, std::vector<int> &labels, int &numLabels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    uint32_t magicNumber;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber); // Convertir en format correct
    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = __builtin_bswap32(numLabels);

    labels.resize(numLabels);
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }

    file.close();
}
