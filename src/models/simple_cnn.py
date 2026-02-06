"""
Modèle CNN simple pour la classification de malaria.
Architecture baseline avec 3 blocs convolutifs.
"""

import torch.nn as nn
from src.config import NOMBRE_CLASSES


class SimpleCNN(nn.Module):
    """Modèle CNN simple pour la classification binaire."""
    
    def __init__(self, taille_image=64):
        super(SimpleCNN, self).__init__()
        self.taille_image = taille_image
        
      
        self.couches_convolution = nn.Sequential(
            # Bloc 1: 3 -> 32 canaux
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            # Bloc 2: 32 -> 64 canaux
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            # Bloc 3: 64 -> 128 canaux
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )
        
        # Calcul de la taille après convolutions
        taille_apres_conv = taille_image // 8  # 3 MaxPool de 2x2
        
        # Couches de classification
        self.couches_classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * taille_apres_conv * taille_apres_conv, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NOMBRE_CLASSES)
        )
    
    def forward(self, tenseur_entree):
        caracteristiques = self.couches_convolution(tenseur_entree)
        sortie = self.couches_classification(caracteristiques)
        return sortie
