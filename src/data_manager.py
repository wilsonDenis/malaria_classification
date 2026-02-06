"""
Gestionnaire de données pour le dataset Malaria.
Chargement, préparation et augmentation des images avec PyTorch.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.config import CHEMIN_DONNEES, CLASSES, TAILLE_IMAGE, TAILLE_BATCH


class MalariaDataset(Dataset):
    """Dataset PyTorch pour les images de cellules sanguines."""
    
    def __init__(self, chemins_images, etiquettes, transformation=None):
        self.chemins_images = chemins_images
        self.etiquettes = etiquettes
        self.transformation = transformation
    
    def __len__(self):
        return len(self.chemins_images)
    
    def __getitem__(self, index):
        chemin_image = self.chemins_images[index]
        image = Image.open(chemin_image).convert('RGB')
        etiquette = self.etiquettes[index]
        
        if self.transformation:
            image = self.transformation(image)
        
        return image, etiquette


class DataManager:
    """Classe pour gérer le chargement et la préparation des données."""
    
    def __init__(self):
        self.chemins_entrainement = []
        self.etiquettes_entrainement = []
        self.chemins_validation = []
        self.etiquettes_validation = []
        self.chemins_test = []
        self.etiquettes_test = []
        
    def load_data(self):
        """Charge les chemins d'images et les étiquettes depuis le dossier de données."""
        liste_chemins = []
        liste_etiquettes = []
        
        print("📂 Chargement des images...")
        
        for index_classe, nom_classe in enumerate(CLASSES):
            chemin_classe = os.path.join(CHEMIN_DONNEES, nom_classe)
            
            if not os.path.exists(chemin_classe):
                print(f"Dossier non trouvé: {chemin_classe}")
                continue
            
            fichiers_images = [f for f in os.listdir(chemin_classe) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"   {nom_classe}: {len(fichiers_images)} images")
                
            for nom_fichier in fichiers_images:
                chemin_complet = os.path.join(chemin_classe, nom_fichier)
                liste_chemins.append(chemin_complet)
                liste_etiquettes.append(index_classe)
        
        
        chemins_train, chemins_temp, etiquettes_train, etiquettes_temp = train_test_split(
            liste_chemins, liste_etiquettes, 
            test_size=0.3, random_state=42, stratify=liste_etiquettes
        )
        
        chemins_val, chemins_test, etiquettes_val, etiquettes_test = train_test_split(
            chemins_temp, etiquettes_temp, 
            test_size=0.5, random_state=42, stratify=etiquettes_temp
        )
        
        self.chemins_entrainement = chemins_train
        self.etiquettes_entrainement = etiquettes_train
        self.chemins_validation = chemins_val
        self.etiquettes_validation = etiquettes_val
        self.chemins_test = chemins_test
        self.etiquettes_test = etiquettes_test
        
        print(f"\n Dataset chargé:")
        print(f"   - Entraînement: {len(chemins_train)} images")
        print(f"   - Validation: {len(chemins_val)} images")
        print(f"   - Test: {len(chemins_test)} images")
        
        return self
    
    def get_transformation_entrainement(self):
        """Retourne les transformations avec augmentation pour l'entraînement."""
        return transforms.Compose([
            transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_transformation_test(self):
        """Retourne les transformations de base pour test/validation."""
        return transforms.Compose([
            transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_dataloaders(self, avec_augmentation=True):
        """Crée et retourne les DataLoaders pour train, val et test."""
        
        transformation_train = self.get_transformation_entrainement() if avec_augmentation else self.get_transformation_test()
        transformation_test = self.get_transformation_test()
        
        dataset_train = MalariaDataset(
            self.chemins_entrainement, 
            self.etiquettes_entrainement, 
            transformation_train
        )
        dataset_val = MalariaDataset(
            self.chemins_validation, 
            self.etiquettes_validation, 
            transformation_test
        )
        dataset_test = MalariaDataset(
            self.chemins_test, 
            self.etiquettes_test, 
            transformation_test
        )
        
        chargeur_train = DataLoader(dataset_train, batch_size=TAILLE_BATCH, shuffle=True)
        chargeur_val = DataLoader(dataset_val, batch_size=TAILLE_BATCH, shuffle=False)
        chargeur_test = DataLoader(dataset_test, batch_size=TAILLE_BATCH, shuffle=False)
        
        return chargeur_train, chargeur_val, chargeur_test
