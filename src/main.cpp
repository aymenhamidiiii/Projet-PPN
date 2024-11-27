#include "Conv2D.h"
#include "MaxPooling2D.h"
#include "MNISTReader.h"
#include "Flatten.h"
#include "Dense.h"
#include "LossFunctions.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Charger les données MNIST
    std::vector<std::vector<double>> trainImages;
    std::vector<int> trainLabels;
    std::vector<std::vector<double>> testImages;
    std::vector<int> testLabels;

    int numTrainImages, trainImageSize, numTrainLabels;
    int numTestImages, testImageSize, numTestLabels;

    readMNISTImages("data/train-images.idx3-ubyte", trainImages, numTrainImages, trainImageSize);
    readMNISTLabels("data/train-labels.idx3-ubyte", trainLabels, numTrainLabels);
    readMNISTImages("data/t10k-images.idx3-ubyte", testImages, numTestImages, testImageSize);
    readMNISTLabels("data/t10k-labels.idx3-ubyte", testLabels, numTestLabels);

    // Vérifier les correspondances
    if (numTrainImages != numTrainLabels || numTestImages != numTestLabels) {
        std::cerr << "Mismatch between image and label counts!" << std::endl;
        return 1;
    }

    std::cout << "Training Data: " << numTrainImages << " images, " << trainImageSize << " pixels each." << std::endl;
    std::cout << "Test Data: " << numTestImages << " images, " << testImageSize << " pixels each." << std::endl;

    // Préparer les données d'entrée et les labels pour l'entraînement
    std::vector<std::vector<std::vector<std::vector<double>>>> trainData(
        numTrainImages, std::vector<std::vector<std::vector<double>>>(1,
            std::vector<std::vector<double>>(28, std::vector<double>(28, 0.0))));

    std::vector<std::vector<double>> trainOutputs(numTrainImages, std::vector<double>(10, 0.0));

    for (int i = 0; i < numTrainImages; ++i) {
        trainOutputs[i][trainLabels[i]] = 1.0;
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                trainData[i][0][y][x] = trainImages[i][y * 28 + x] / 255.0; // Normalisation
            }
        }
    }

    // Définir les couches du réseau
    Conv2D conv1(30, 5, 1, 28, 28, 1, "valid");
    MaxPooling2D maxPool1(2, 2, "valid");
    Conv2D conv2(15, 3, 1, 12, 12, 30, "valid");
    MaxPooling2D maxPool2(2, 2, "valid");
    Flatten flatten;
    Dense dense1(375, 128, "relu");
    Dense dense2(128, 50, "relu");
    Dense outputLayer(50, 10, "softmax");

    // Paramètres d'entraînement
    const int epochs = 5;
    const double learningRate = 0.001;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        // Étape 1 : Entraînement
        for (size_t i = 0; i < trainData.size(); ++i) {
            auto conv1Output = conv1.forward(trainData[i]);
            auto maxPool1Output = maxPool1.forward(conv1Output);
            auto conv2Output = conv2.forward(maxPool1Output);
            auto maxPool2Output = maxPool2.forward(conv2Output);
            auto flattenOutput = flatten.forward(maxPool2Output);
            auto dense1Output = dense1.forward(flattenOutput);
            auto dense2Output = dense2.forward(dense1Output);
            auto finalOutput = outputLayer.forward(dense2Output);

            const std::vector<double>& trueLabel = trainOutputs[i];
            double loss = computeCrossEntropyLoss(finalOutput, trueLabel);
            totalLoss += loss;

            auto dLoss_dOutput = computeSoftmaxGradient(finalOutput, trueLabel);
            auto dLoss_dDense2 = outputLayer.backward(dLoss_dOutput, dense2Output, learningRate);
            auto dLoss_dDense1 = dense2.backward(dLoss_dDense2, dense1Output, learningRate);
            dense1.backward(dLoss_dDense1, flattenOutput, learningRate);
        }

        // Étape 2 : Évaluation sur l'ensemble de test
        int correctPredictions = 0;
        for (size_t i = 0; i < testImages.size(); ++i) {
            std::vector<std::vector<std::vector<std::vector<double>>>> singleTestData(
                1, std::vector<std::vector<std::vector<double>>>(1,
                    std::vector<std::vector<double>>(28, std::vector<double>(28, 0.0))));
            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    singleTestData[0][0][y][x] = testImages[i][y * 28 + x] / 255.0;
                }
            }

            auto conv1Output = conv1.forward(singleTestData[0]);
            auto maxPool1Output = maxPool1.forward(conv1Output);
            auto conv2Output = conv2.forward(maxPool1Output);
            auto maxPool2Output = maxPool2.forward(conv2Output);
            auto flattenOutput = flatten.forward(maxPool2Output);
            auto dense1Output = dense1.forward(flattenOutput);
            auto dense2Output = dense2.forward(dense1Output);
            auto finalOutput = outputLayer.forward(dense2Output);

            int predictedClass = std::distance(finalOutput.begin(), std::max_element(finalOutput.begin(), finalOutput.end()));
            if (predictedClass == testLabels[i]) {
                ++correctPredictions;
            }
        }

        double accuracy = static_cast<double>(correctPredictions) / testImages.size();
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / trainData.size()
                  << ", Accuracy: " << accuracy * 100 << "%" << std::endl;
    }

    return 0;
}
