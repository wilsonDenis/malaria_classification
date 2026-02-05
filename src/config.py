import os
import torch

TAILLE_IMAGE = 64
TAILLE_BATCH = 32
NOMBRE_EPOCHS = 30
TAUX_APPRENTISSAGE = 0.001
PATIENCE = 5

CLASSES = ['Parasitee', 'Non_parasitee']
NOMBRE_CLASSES = len(CLASSES)

CHEMIN_DONNEES = './data/malaria_hematie_dataset'
CHEMIN_RESULTATS = './resultats'

os.makedirs(CHEMIN_RESULTATS, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("GPU NVIDIA détecté")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("GPU Apple Silicon (MPS) détecté")
else:
    DEVICE = torch.device('cpu')
    print("Utilisation du CPU")
