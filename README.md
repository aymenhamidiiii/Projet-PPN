# Projet PPN - Reconnaissance de Chiffres Manuscrits

Ce projet implémente un modèle de réseau de neurones convolutif (CNN) pour reconnaître des chiffres manuscrits à partir du dataset MNIST. L'architecture du modèle est optimisée pour traiter des images 28x28 en niveaux de gris.

---

## 🚀 Étapes pour exécuter le projet

### 1. Cloner le projet
```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-projet>.git
cd <nom-du-projet>


## Structure

- `src/` : Code source
- `include/` : Fichiers d'en-tête
- `build/` : Fichiers de construction CMake

### 2. Préparer l'environnement
Assurez-vous que votre système dispose des éléments suivants :

- Un compilateur C++ (GCC ou Clang recommandé).
- Les outils de construction CMake et Make.
- Les bibliothèques standard nécessaires à la compilation. 

### Compiler le projet 
Créez un répertoire dédié à la compilation et compilez le projet :

mkdir build
cd build
cmake ..
make

### 4. Télécharger le dataset MNIST 

Téléchargez les fichiers nécessaires depuis le site officiel MNIST et placez-les dans un dossier data à la racine du projet :

train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte

### Exécuter le programme
Lancez le programme via le binaire généré :
./projetPPN

## Architecture du modèle 
L'architecture du réseau est organisée en plusieurs étapes :
### 1. Couches Convolutives (Conv2D) :
Première couche : 30 filtres de taille 5x5, stride 1, valid padding.
Deuxième couche : 15 filtres de taille 3x3, stride 1, valid padding.

### 2. Couches Max-Pooling (MaxPooling2D) :
Réduction de dimension avec une fenêtre de 2x2 après chaque couche convolutive.

### 3. Couche de Flattening (Flatten) :
Transforme les sorties multidimensionnelles en un vecteur 1D.

### 4. Couches Denses (Dense) :
Dense1 : 375 entrées → 128 sorties, activation ReLU.
Dense2 : 128 entrées → 50 sorties, activation ReLU.
OutputLayer : 50 entrées → 10 sorties (classes), activation Softmax.

## Exemple d'output : 
Training Data: 60000 images, 784 pixels each.
Test Data: 10000 images, 784 pixels each.

aymen@DESKTOP-1OMIBGS:/mnt/c/Users/HP/Desktop/Nouveau/Projet-PPN/build$ ./projetPPN
Training Data: 60000 images, 784 pixels each.
Test Data: 10000 images, 784 pixels each.
Epoch 1, Loss: 0.0734936, Training Accuracy: 79.96%, Test Accuracy: 86.82%
Epoch 2, Loss: 0.0355405, Training Accuracy: 89.195%, Test Accuracy: 89.54%
Epoch 3, Loss: 0.0289288, Training Accuracy: 91.1517%, Test Accuracy: 91.09%
Epoch 4, Loss: 0.0251219, Training Accuracy: 92.345%, Test Accuracy: 92.25%
Epoch 5, Loss: 0.0225264, Training Accuracy: 93.1%, Test Accuracy: 92.95%


