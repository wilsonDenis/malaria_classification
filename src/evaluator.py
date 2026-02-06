"""
Évaluateur de modèles pour la classification de malaria.
Génère les métriques et visualisations.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.config import APPAREIL, CLASSES, CHEMIN_RESULTATS


class Evaluator:

    
    def __init__(self, modele, nom_modele):
        self.modele = modele
        self.nom_modele = nom_modele
        self.modele.to(APPAREIL)
        
    def evaluate(self, chargeur_test):
      
        self.modele.eval()
        
        toutes_predictions = []
        toutes_etiquettes = []
        
        with torch.no_grad():
            for images_batch, etiquettes_batch in chargeur_test:
                images_batch = images_batch.to(APPAREIL)
                
                sorties = self.modele(images_batch)
                _, predictions = torch.max(sorties, 1)
                
                toutes_predictions.extend(predictions.cpu().numpy())
                toutes_etiquettes.extend(etiquettes_batch.numpy())
        
        tableau_predictions = np.array(toutes_predictions)
        tableau_etiquettes = np.array(toutes_etiquettes)
        
        
        matrice_confusion = confusion_matrix(tableau_etiquettes, tableau_predictions)
        
     
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            matrice_confusion, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            annot_kws={'size': 14}
        )
        plt.title(f'Matrice de Confusion - {self.nom_modele}', fontsize=14, fontweight='bold')
        plt.ylabel('Vraie classe', fontsize=12)
        plt.xlabel('Classe prédite', fontsize=12)
        
        chemin_figure = os.path.join(CHEMIN_RESULTATS, f'{self.nom_modele}_confusion.png')
        plt.savefig(chemin_figure, dpi=150, bbox_inches='tight')
        plt.close()
        
       
        precision_globale = np.mean(tableau_predictions == tableau_etiquettes) * 100
        

        print(f"\n{'='*60}")
        print(f" RÉSULTATS - {self.nom_modele}")
        print(f"{'='*60}")
        print(f"   Précision globale: {precision_globale:.2f}%")
        print(f"{'='*60}")
        print(classification_report(tableau_etiquettes, tableau_predictions, target_names=CLASSES))
        
        rapport = classification_report(
            tableau_etiquettes, 
            tableau_predictions, 
            target_names=CLASSES, 
            output_dict=True
        )
        
        return rapport
