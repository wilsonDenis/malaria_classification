📓 **Notebook Jupyter - Classification Malaria avec PyTorch**

## Utilisation

```bash
cd /Users/wilson-bahun/Downloads/malaria_classification
jupyter notebook classification_malaria_complete.ipynb
```

## Contenu du Notebook

Le notebook contient tout le rendu du projet:

1. **Configuration** - Device PyTorch, hyperparamètres
2. **Chargement des données** - Avec visualisation d'échantillons
3. **Préparation** - Augmentation et DataLoaders
4. **Modèles** - CNN Simple, VGG16, ResNet50
5. **Entraînement** - Avec courbes d'apprentissage pour chaque modèle
6. **Évaluation** - Matrices de confusion et courbes ROC
7. **Comparaison** - Graphique comparatif final

## Variables Significatives

Toutes les variables ont des noms en français compréhensibles:
- `taille_image`, `nombre_epochs`, `taux_apprentissage`
- `chemins_train`, `labels_train`, `dataset_train`
- `historique_train_loss`, `historique_val_acc`
- `toutes_predictions`, `tous_labels`, `matrice_confusion`

## Visualisations Incluses

- ✅ Exemples d'images du dataset
- ✅ Courbes d'apprentissage (Loss + Accuracy) pour chaque modèle
- ✅ Matrices de confusion pour chaque modèle
- ✅ Courbes ROC avec AUC pour chaque modèle  
- ✅ Graphique de comparaison finale

Toutes les visualisations sont sauvegardées dans `./resultats/`
