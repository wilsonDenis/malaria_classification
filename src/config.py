"""
Configuration globale du projet de classification de malaria.
Contient tous les paramètres et constantes.
"""

import os
import torch


TAILLE_IMAGE = 64
TAILLE_BATCH = 32


NOMBRE_EPOCHS = 15
TAUX_APPRENTISSAGE = 0.001
PATIENCE = 5


CLASSES = ['parasitized', 'uninfected']
NOMBRE_CLASSES = len(CLASSES)


CHEMIN_DONNEES = './data/malaria_hematie_dataset'
CHEMIN_RESULTATS = './resultats'

os.makedirs(CHEMIN_RESULTATS, exist_ok=True)


def obtenir_appareil():
    """Détecte le meilleur appareil disponible (GPU/CPU)."""
    if torch.cuda.is_available():
        appareil = torch.device('cuda')
        print(f" GPU NVIDIA détecté: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        appareil = torch.device('mps')
        print(" GPU Apple Silicon (MPS) détecté")
    else:
        appareil = torch.device('cpu')
        print(" Utilisation du CPU")
    return appareil

APPAREIL = obtenir_appareil()
