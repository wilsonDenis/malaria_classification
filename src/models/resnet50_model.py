"""
Modèle ResNet50 pré-entraîné pour la classification de malaria.
Utilise le transfer learning avec fine-tuning.
"""

import torch.nn as nn
from torchvision import models
from src.config import NOMBRE_CLASSES


class ResNet50Model(nn.Module):
    """Modèle ResNet50 avec transfer learning."""
    
    def __init__(self, taille_image=64):
        super(ResNet50Model, self).__init__()
        self.taille_image = taille_image
        
        # Chargement du modèle pré-entraîné
        modele_resnet = models.resnet50(weights='DEFAULT')
        
        # Gel de toutes les couches sauf layer4
        for nom_couche, couche in modele_resnet.named_children():
            if nom_couche != 'layer4' and nom_couche != 'fc':
                for parametre in couche.parameters():
                    parametre.requires_grad = False
        
        # Récupération du nombre de features avant la couche FC
        nombre_features_entree = modele_resnet.fc.in_features
        
        # Remplacement de la couche FC
        modele_resnet.fc = nn.Sequential(
            nn.Linear(nombre_features_entree, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NOMBRE_CLASSES)
        )
        
        self.modele_complet = modele_resnet
    
    def forward(self, tenseur_entree):
        sortie = self.modele_complet(tenseur_entree)
        return sortie
