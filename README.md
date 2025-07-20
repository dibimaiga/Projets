# Classificateur Fashion-MNIST

Une implémentation PyTorch de deux architectures complémentaires - 
un réseau neuronal convolutif (CNNet) et un perceptron multicouche (MLP) entièrement connecté - formées à partir de zéro sur l'ensemble de données [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

## CNNet 
  - 2 × Conv (5×5) → ReLU → MaxPool  
  - 3 × couches linéaires (100 → 60 → 10) avec activations ReLU:contentReference[oaicite:0]{index=0}  
    ## MLP   
  - Aplatir 28×28 → Linéaire 784→300 → ReLU → Linéaire 300→10:contentReference[oaicite:1]{index=1}  
  Les pipelines d'entrée normalisent les images sur [-1, 1] et diffusent des mini-lots de 32 échantillons:contentReference[oaicite:2]{index=2}.  
  Entraîné pendant 10 époques avec SGD (0,01 lr, 0,9 momentum), atteignant ≈ 88-89 % de précision de test sur une seule boîte GPU/CPU:contentReference[oaicite:3]{index=3}.

