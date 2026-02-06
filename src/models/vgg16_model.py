"""
Modèle VGG16 pré-entraîné pour la classification de malaria.
Utilise le transfer learning avec fine-tuning.
"""

import torch.nn as nn
from torchvision import models
from src.config import NOMBRE_CLASSES


class VGG16Model(nn.Module):
    """Modèle VGG16 avec transfer learning."""
    
    def __init__(self, taille_image=64):
        super(VGG16Model, self).__init__()
        self.taille_image = taille_image
        
        # Chargement du modèle pré-entraîné
        modele_vgg = models.vgg16(weights='DEFAULT')
        
        # Gel des premières couches
        for couche in modele_vgg.features[:-6]:
            for parametre in couche.parameters():
                parametre.requires_grad = False
        
        self.extracteur_caracteristiques = modele_vgg.features
        
     
        taille_sortie = taille_image // 32 
        
        
        self.couches_classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * taille_sortie * taille_sortie, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NOMBRE_CLASSES)
        )
    
    def forward(self, tenseur_entree):
        caracteristiques = self.extracteur_caracteristiques(tenseur_entree)
        sortie = self.couches_classification(caracteristiques)
        return sortie
