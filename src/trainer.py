"""
Entraîneur de modèles pour la classification de malaria.
Gère l'entraînement avec early stopping et sauvegarde du meilleur modèle.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import APPAREIL, TAUX_APPRENTISSAGE, NOMBRE_EPOCHS, PATIENCE, CHEMIN_RESULTATS


class Trainer:
    """Classe pour entraîner les modèles de classification."""
    
    def __init__(self, modele, nom_modele):
        self.modele = modele
        self.nom_modele = nom_modele
        self.modele.to(APPAREIL)
        
    def train(self, chargeur_entrainement, chargeur_validation):
        """
        Entraîne le modèle avec les données fournies.
        
        Args:
            chargeur_entrainement: DataLoader d'entraînement
            chargeur_validation: DataLoader de validation
        
        Returns:
            historique: Dictionnaire avec les métriques d'entraînement
        """
        
        critere = nn.CrossEntropyLoss()
        optimiseur = optim.Adam(self.modele.parameters(), lr=TAUX_APPRENTISSAGE)
        planificateur = optim.lr_scheduler.ReduceLROnPlateau(
            optimiseur, mode='min', factor=0.5, patience=3
        )
        
        historique = {
            'perte_entrainement': [],
            'precision_entrainement': [],
            'perte_validation': [],
            'precision_validation': []
        }
        
        meilleure_perte_validation = float('inf')
        compteur_patience = 0
        
        print(f"\n{'='*60}")
        print(f"ENTRAÎNEMENT - {self.nom_modele}")
        print(f"{'='*60}")
        print(f"   Époques max: {NOMBRE_EPOCHS}")
        print(f"   Taux apprentissage: {TAUX_APPRENTISSAGE}")
        print(f"   Appareil: {APPAREIL}")
        print(f"{'='*60}\n")
        
        for epoque in range(NOMBRE_EPOCHS):
         
            self.modele.train()
            perte_totale_train = 0.0
            predictions_correctes_train = 0
            total_echantillons_train = 0
            
            barre_progression = tqdm(chargeur_entrainement, desc=f'Époque {epoque+1}/{NOMBRE_EPOCHS}')
            
            for images_batch, etiquettes_batch in barre_progression:
                images_batch = images_batch.to(APPAREIL)
                etiquettes_batch = etiquettes_batch.to(APPAREIL)
                
                optimiseur.zero_grad()
                sorties = self.modele(images_batch)
                perte = critere(sorties, etiquettes_batch)
                perte.backward()
                optimiseur.step()
                
                perte_totale_train += perte.item() * images_batch.size(0)
                _, predictions = torch.max(sorties, 1)
                predictions_correctes_train += (predictions == etiquettes_batch).sum().item()
                total_echantillons_train += etiquettes_batch.size(0)
                
                barre_progression.set_postfix({'perte': f'{perte.item():.4f}'})
            
            perte_moyenne_train = perte_totale_train / total_echantillons_train
            precision_train = 100 * predictions_correctes_train / total_echantillons_train
            
           
            perte_validation, precision_validation = self._evaluer(chargeur_validation, critere)
            
         
            planificateur.step(perte_validation)
            
          
            historique['perte_entrainement'].append(perte_moyenne_train)
            historique['precision_entrainement'].append(precision_train)
            historique['perte_validation'].append(perte_validation)
            historique['precision_validation'].append(precision_validation)
            
            print(f"Époque [{epoque+1:2d}/{NOMBRE_EPOCHS}] "
                  f"Train: {precision_train:.1f}% | Val: {precision_validation:.1f}%")
            
           
            if perte_validation < meilleure_perte_validation:
                meilleure_perte_validation = perte_validation
                compteur_patience = 0
                chemin_sauvegarde = os.path.join(CHEMIN_RESULTATS, f'{self.nom_modele}_meilleur.pth')
                torch.save(self.modele.state_dict(), chemin_sauvegarde)
                print(f"   Meilleur modèle sauvegardé!")
            else:
                compteur_patience += 1
                if compteur_patience >= PATIENCE:
                    print(f"\n Early stopping après {epoque+1} époques")
                    break
        
     
        chemin_meilleur = os.path.join(CHEMIN_RESULTATS, f'{self.nom_modele}_meilleur.pth')
        self.modele.load_state_dict(torch.load(chemin_meilleur, map_location=APPAREIL))
        
        print(f"\n{'='*60}")
        print(f"✅ Entraînement terminé!")
        print(f"   Meilleure précision validation: {max(historique['precision_validation']):.1f}%")
        print(f"{'='*60}\n")
        
        return historique
    
    def _evaluer(self, chargeur_donnees, critere):
        """Évalue le modèle sur un ensemble de données."""
        self.modele.eval()
        perte_totale = 0.0
        predictions_correctes = 0
        total_echantillons = 0
        
        with torch.no_grad():
            for images_batch, etiquettes_batch in chargeur_donnees:
                images_batch = images_batch.to(APPAREIL)
                etiquettes_batch = etiquettes_batch.to(APPAREIL)
                
                sorties = self.modele(images_batch)
                perte = critere(sorties, etiquettes_batch)
                
                perte_totale += perte.item() * images_batch.size(0)
                _, predictions = torch.max(sorties, 1)
                predictions_correctes += (predictions == etiquettes_batch).sum().item()
                total_echantillons += etiquettes_batch.size(0)
        
        perte_moyenne = perte_totale / total_echantillons
        precision = 100 * predictions_correctes / total_echantillons
        
        return perte_moyenne, precision
