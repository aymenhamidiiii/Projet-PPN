Projet PPN - Reconnaissance de Chiffres Manuscrits
Ce projet implémente un modèle de réseau de neurones convolutif (CNN) pour la reconnaissance de chiffres manuscrits en utilisant le dataset MNIST. L'architecture est optimisée pour traiter des images de taille 28x28 en niveaux de gris.

🚀 Étapes pour exécuter le projet
1. Cloner le projet
bash
Copier le code
git clone https://github.com/<votre-utilisateur>/<nom-du-projet>.git
cd <nom-du-projet>

2. Préparer l'environnement
Assurez-vous que votre système est configuré avec les éléments suivants :

Un compilateur C++ (GCC ou Clang recommandé).
Les outils de construction : CMake et Make.
Les bibliothèques standard nécessaires à la compilation.

3. Compiler le projet
Créez un répertoire dédié à la compilation et générez les fichiers exécutables :

bash
Copier le code
mkdir build
cd build
cmake ..
make

4. Télécharger le dataset MNIST
Téléchargez les fichiers nécessaires depuis le site officiel de MNIST et placez-les dans un dossier data à la racine du projet :

train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte

5. Exécuter le programme
Lancez le programme via le binaire généré :

bash
Copier le code
./projetPPN
📂 Structure du projet
src/ : Code source principal.
include/ : Fichiers d'en-tête.
build/ : Fichiers générés pour la construction via CMake.
data/ : Dataset MNIST.
🧠 Architecture du modèle
L'architecture du réseau CNN est organisée comme suit :

Couches Convolutives (Conv2D) :

Première couche : 30 filtres de taille 5x5, stride 1, valid padding.
Deuxième couche : 15 filtres de taille 3x3, stride 1, valid padding.
Couches Max-Pooling (MaxPooling2D) :

Réduction de la dimension avec une fenêtre de 2x2 après chaque couche convolutive.
Couche de Flattening (Flatten) :

Transformation des sorties multidimensionnelles en un vecteur 1D.
Couches Denses (Dense) :

Dense 1 : 375 entrées → 128 sorties, activation ReLU.
Dense 2 : 128 entrées → 50 sorties, activation ReLU.
Couche de sortie : 50 entrées → 10 sorties (correspondant aux classes de chiffres), activation Softmax.
📊 Exemple de sortie
Voici un exemple d'output lors de l'exécution du programme :

plaintext
Copier le code
Training Data: 60000 images, 784 pixels each.
Test Data: 10000 images, 784 pixels each.
Epoch 1, Loss: 0.0734936, Training Accuracy: 79.96%, Test Accuracy: 86.82%
Epoch 2, Loss: 0.0355405, Training Accuracy: 89.195%, Test Accuracy: 89.54%
Epoch 3, Loss: 0.0289288, Training Accuracy: 91.1517%, Test Accuracy: 91.09%
Epoch 4, Loss: 0.0251219, Training Accuracy: 92.345%, Test Accuracy: 92.25%
Epoch 5, Loss: 0.0225264, Training Accuracy: 93.1%, Test Accuracy: 92.95%

📝 Remarques
Les performances du modèle peuvent varier en fonction des hyperparamètres et de l'initialisation des poids.
Les fichiers générés (exécutables et logs) sont placés dans le répertoire build/.

🔗 Ressources supplémentaires
Site officiel MNIST
Documentation CMake
GCC Compiler
