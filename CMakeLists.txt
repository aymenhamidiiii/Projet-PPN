cmake_minimum_required(VERSION 3.10)

# Nom du projet
project(projetPPN)

# Spécifier la version de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Activer les avertissements et l'optimisation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -O3")

# Ajouter les fichiers sources
set(SOURCES
    src/main.cpp
    src/Conv2D.cpp
    src/MaxPooling2D.cpp
    src/Flatten.cpp
    src/Dense.cpp
    src/Layer.cpp
    src/Neuron.cpp
    src/NeuralNetwork.cpp
    src/MNISTReader.cpp
    src/LossFunctions.cpp
    src/AdamOptimizer.cpp
)


# Inclure le répertoire des fichiers d'en-tête
include_directories(include)

# Générer l'exécutable
add_executable(projetPPN ${SOURCES})

# Afficher un message après configuration
message(STATUS "Configuration terminée. Vous pouvez compiler avec 'make'.")
