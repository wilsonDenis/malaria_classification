# Classification du Paludisme par Deep Learning

## 📋 Description du Projet

Ce projet implémente un système de classification d'images pour détecter automatiquement le paludisme dans des cellules sanguines (hématies) en utilisant des réseaux de neurones convolutifs (CNN). Le paludisme étant une maladie infectieuse grave touchant des millions de personnes chaque année, particulièrement dans les régions tropicales et subtropicales, ce projet vise à accélérer et améliorer la précision du diagnostic.

### Objectif

Développer et entraîner des modèles de Deep Learning capables de différencier les hématies infectées par le parasite du paludisme de celles qui sont saines, offrant ainsi une assistance automatisée aux professionnels de santé.

## 🏗️ Architecture du Projet

```
malaria_classification/
├── data/
│   └── malaria_hematie_dataset/
│       ├── Parasitee/          # Images d'hématies infectées
│       └── Non_parasitee/      # Images d'hématies saines
├── src/
│   ├── config.py               # Configuration globale
│   ├── data_manager.py         # Classe DataManager (chargement et augmentation)
│   ├── trainer.py              # Classe Trainer (entraînement avec callbacks)
│   ├── evaluator.py            # Classe Evaluator (évaluation et métriques)
│   └── models/
│       ├── __init__.py
│       ├── base_model.py       # Classe abstraite BaseModel
│       ├── simple_cnn.py       # Classe SimpleCNN (from scratch)
│       ├── vgg16_model.py      # Classe VGG16Model (fine-tuned)
│       └── resnet50_model.py   # Classe ResNet50Model (fine-tuned)
├── resultats/                  # Résultats d'entraînement
├── main.py                     # Script principal (MalariaClassificationPipeline)
├── requirements.txt
└── README.md
```

## 📊 Structure des Données

### Organisation

Le dataset est organisé en deux catégories:
- **Parasitee**: Hématies infectées par le parasite du paludisme
- **Non_parasitee**: Hématies saines

### Préparation des Données

#### 1. Chargement et Normalisation
Les images sont chargées depuis les dossiers, redimensionnées à 64x64 pixels, et normalisées entre 0 et 1 en divisant par 255.

#### 2. Séparation des Données
- **70%** pour l'entraînement
- **15%** pour la validation
- **15%** pour le test

Cette séparation est effectuée de manière stratifiée pour maintenir la proportion des classes.

#### 3. Augmentation des Données

L'augmentation de données est appliquée uniquement sur l'ensemble d'entraînement pour éviter l'overfitting et améliorer la généralisation. Les transformations incluent:
- **Rotation aléatoire**: jusqu'à 20 degrés
- **Décalage horizontal/vertical**: jusqu'à 20% de la taille de l'image
- **Cisaillement**: jusqu'à 20%
- **Zoom**: jusqu'à 20%
- **Retournement horizontal et vertical**: pour simuler différentes orientations

## 🧠 Modèles Implémentés

### 1. CNN Simple (From Scratch) - Classe `SimpleCNN`

**Classe:** `SimpleCNN` hérite de `BaseModel`

**Architecture:**
- **Bloc 1**: Conv2D(32 filtres, 3x3) → MaxPooling(2x2) → BatchNormalization
- **Bloc 2**: Conv2D(64 filtres, 3x3) → MaxPooling(2x2) → BatchNormalization
- **Bloc 3**: Conv2D(128 filtres, 3x3) → MaxPooling(2x2) → BatchNormalization
- **Classifieur**: Flatten → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(2, softmax)

**Caractéristiques:**
- Modèle construit entièrement from scratch
- BatchNormalization pour stabiliser l'entraînement
- Dropout pour réduire l'overfitting
- Activation ReLU pour introduire la non-linéarité

### 2. VGG16 Fine-tuned - Classe `VGG16Model`

**Classe:** `VGG16Model` hérite de `BaseModel`

**Architecture:**
- **Encodeur**: VGG16 pré-entraîné sur ImageNet (sans les couches de classification)
- **Gel**: Toutes les couches sauf les 4 dernières blocs sont gelées
- **Classifieur personnalisé**: Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(2, softmax)

**Prétraitement spécifique:**
Les images sont prétraitées avec `preprocess_input` de VGG16 qui applique la normalisation standard utilisée lors de l'entraînement sur ImageNet (soustraction de la moyenne des canaux RGB).

**Avantages:**
- Utilise des features pré-apprises sur un large dataset
- Nécessite moins de données d'entraînement
- Converge plus rapidement

### 3. ResNet50 Fine-tuned - Classe `ResNet50Model`

**Classe:** `ResNet50Model` hérite de `BaseModel`

**Architecture:**
- **Encodeur**: ResNet50 pré-entraîné sur ImageNet (sans les couches de classification)
- **Gel**: Toutes les couches sauf les 10 dernières sont gelées
- **Classifieur personnalisé**: GlobalAveragePooling2D → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(2, softmax)

**Prétraitement spécifique:**
Les images sont prétraitées avec `preprocess_input` de ResNet50.

**Avantages:**
- Architecture avec connexions résiduelles (skip connections)
- Évite le problème de gradient vanishing
- Très performant pour la classification d'images

## 🏛️ Architecture POO (Programmation Orientée Objet)

Le projet utilise une architecture orientée objet avec plusieurs classes principales:

### Classe `BaseModel` (Abstraite)
Classe de base pour tous les modèles CNN avec méthodes communes:
- `build()`: Construction du modèle (méthode abstraite)
- `compile_model()`: Compilation avec optimizer Adam
- `get_model()`: Récupération du modèle Keras
- `save_weights()`, `load_weights()`: Sauvegarde/chargement des poids

### Classe `DataManager`
Gestion complète des données:
- `load_images_from_folders()`: Chargement depuis dossiers
- `prepare_data()`: Split train/val/test avec stratification
- `get_data_generator()`: Création du générateur d'augmentation
- `show_augmentation_examples()`: Visualisation des augmentations
- `get_dataset_info()`: Informations sur le dataset

### Classe `Trainer`
Entraînement des modèles avec callbacks:
- `create_callbacks()`: Création EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- `train()`: Boucle d'entraînement complète
- `save_learning_curves()`: Sauvegarde des courbes

### Classe `Evaluator`
Évaluation et visualisation:
- `evaluate()`: Calcul de toutes les métriques
- `print_results()`: Affichage des résultats
- `plot_confusion_matrix()`: Matrice de confusion
- `plot_roc_curve()`: Courbe ROC avec AUC
- `compare_models()`: Comparaison multi-modèles (méthode statique)

### Classe `MalariaClassificationPipeline`
Orchestration complète du projet:
- `load_data()`: Chargement via DataManager
- `create_models()`: Instanciation des 3 modèles
- `train_all_models()`: Entraînement avec Trainer
- `evaluate_all_models()`: Évaluation avec Evaluator
- `compare_models()`: Comparaison finale
- `run()`: Exécution complète du pipeline

## 🎯 Entraînement

### Hyperparamètres

```python
TAILLE_IMAGE = 64
TAILLE_BATCH = 32
NOMBRE_EPOCHS = 30
TAUX_APPRENTISSAGE = 0.001
PATIENCE = 5 (pour early stopping)
```

### Callbacks Utilisés

#### 1. Early Stopping
- **Surveillance**: Validation loss
- **Patience**: 5 epochs
- **Fonction**: Arrête l'entraînement si la loss de validation ne s'améliore pas pendant 5 epochs consécutifs
- **Restauration**: Restaure les poids du meilleur epoch

#### 2. ReduceLROnPlateau
- **Surveillance**: Validation loss
- **Facteur de réduction**: 0.5 (divise le learning rate par 2)
- **Patience**: 3 epochs
- **Learning rate minimum**: 1e-7
- **Fonction**: Réduit le learning rate lorsque la performance stagne, permettant une optimisation plus fine

#### 3. ModelCheckpoint
- **Surveillance**: Validation accuracy
- **Sauvegarde**: Uniquement les poids du meilleur modèle
- **Format**: fichier `.weights.h5`

### Optimiseur

L'optimiseur **Adam** est utilisé avec un learning rate initial de 0.001. Adam combine les avantages de:
- **AdaGrad**: Adaptation du learning rate pour chaque paramètre
- **RMSprop**: Utilisation de moyennes mobiles des gradients

### Fonction de Perte

**Categorical Crossentropy** est utilisée car nous avons une classification multi-classes (2 classes: Parasitée et Non parasitée).

## 📈 Évaluation

### Métriques Calculées

#### 1. Accuracy (Exactitude)
Proportion de prédictions correctes sur le total.
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### 2. Precision (Précision)
Proportion de vrais positifs parmi les prédictions positives.
```
Precision = TP / (TP + FP)
```
Utile quand le coût d'un faux positif est élevé.

#### 3. Recall (Rappel / Sensibilité)
Proportion de vrais positifs parmi tous les cas réellement positifs.
```
Recall = TP / (TP + FN)
```
Utile quand le coût d'un faux négatif est élevé (crucial en médecine).

#### 4. F1-Score
Moyenne harmonique de la précision et du rappel.
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
Équilibre entre précision et rappel.

#### 5. Sensibilité
Identique au Recall. Mesure la capacité à détecter les cas positifs (hématies infectées).

#### 6. Spécificité
Proportion de vrais négatifs parmi tous les cas réellement négatifs.
```
Spécificité = TN / (TN + FP)
```
Mesure la capacité à identifier correctement les cas négatifs (hématies saines).

### Visualisations Générées

#### 1. Courbes d'Apprentissage
Pour chaque modèle, ces graphiques montrent:
- **Loss**: Évolution de la perte sur train et validation
- **Accuracy**: Évolution de l'exactitude sur train et validation

Permettent de détecter l'overfitting (écart croissant entre train et validation).

#### 2. Matrice de Confusion
Tableau croisé montrant:
- **Vrais Positifs (TP)**: Parasitée prédite correctement
- **Vrais Négatifs (TN)**: Non parasitée prédite correctement
- **Faux Positifs (FP)**: Non parasitée prédite comme Parasitée
- **Faux Négatifs (FN)**: Parasitée prédite comme Non parasitée

#### 3. Courbe ROC et AUC
- **ROC (Receiver Operating Characteristic)**: Trace le taux de vrais positifs vs taux de faux positifs
- **AUC (Area Under Curve)**: Aire sous la courbe ROC
  - AUC = 1.0: Classifieur parfait
  - AUC = 0.5: Classifieur aléatoire
  - AUC > 0.8: Bon classifieur

#### 4. Graphique de Comparaison
Compare les 6 métriques pour les 3 modèles sur un même graphique, permettant d'identifier rapidement le meilleur modèle.

## 🚀 Utilisation

### Installation

```bash
# Cloner le projet
cd malaria_classification

# Créer un environnement virtuel
python3 -m venv env
source env/bin/activate  # Sur Mac/Linux
# env\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Préparation des Données

Organisez vos images dans la structure suivante:
```
data/malaria_hematie_dataset/
├── Parasitee/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Non_parasitee/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Exécution

```bash
# Lancer l'entraînement complet des 3 modèles
python3 main.py
```

Le script va:
1. Charger et préparer les données
2. Afficher des exemples d'augmentation
3. Entraîner le CNN Simple
4. Entraîner VGG16
5. Entraîner ResNet50
6. Évaluer chaque modèle sur le test set
7. Générer toutes les visualisations
8. Comparer les performances

### Résultats

Tous les résultats sont sauvegardés dans le dossier `resultats/`:
- `CNN_Simple.weights.h5` - Poids du modèle CNN simple
- `VGG16.weights.h5` - Poids du modèle VGG16
- `ResNet50.weights.h5` - Poids du modèle ResNet50
- `CNN_Simple_courbes.png` - Courbes d'apprentissage
- `VGG16_courbes.png` - Courbes d'apprentissage
- `ResNet50_courbes.png` - Courbes d'apprentissage
- `CNN_Simple_matrice_confusion.png` - Matrice de confusion
- `VGG16_matrice_confusion.png` - Matrice de confusion
- `ResNet50_matrice_confusion.png` - Matrice de confusion
- `CNN_Simple_courbe_roc.png` - Courbe ROC
- `VGG16_courbe_roc.png` - Courbe ROC
- `ResNet50_courbe_roc.png` - Courbe ROC
- `comparaison_modeles.png` - Comparaison des 3 modèles
- `augmentation_exemples.png` - Exemples d'augmentation de données

## 🔍 Interprétation des Résultats

### Choix du Meilleur Modèle

Le meilleur modèle dépend du contexte d'utilisation:

**Pour un dépistage médical:**
- Privilégier le **Recall/Sensibilité** élevé (minimiser les faux négatifs)
- Un faux négatif (ne pas détecter un malade) est plus grave qu'un faux positif
- Regarder l'AUC pour la performance globale

**Pour un système de confirmation:**
- Privilégier la **Précision** élevée (minimiser les faux positifs)
- Éviter d'alarmer inutilement avec des faux positifs

**Pour un équilibre:**
- Choisir le modèle avec le meilleur **F1-Score**
- Ou le meilleur **AUC**

### Lecture de la Matrice de Confusion

```
                 Prédiction
              Parasitée  Non_parasitée
Réalité
Parasitée        TP          FN
Non_parasitée    FP          TN
```

**Cas idéal:** Diagonale principale élevée (TP et TN), hors diagonale faible (FP et FN).

### Lecture de la Courbe ROC

- Courbe proche du coin supérieur gauche = Bon modèle
- Courbe sur la diagonale = Modèle aléatoire
- AUC élevé (> 0.9) = Excellent modèle pour la classification

## 🛠️ Technologies Utilisées

- **TensorFlow/Keras**: Framework de Deep Learning
- **NumPy**: Calculs numériques
- **Pandas**: Manipulation de données
- **Scikit-learn**: Métriques et preprocessing
- **Matplotlib/Seaborn**: Visualisations
- **Pillow**: Traitement d'images

## 📝 Méthodologie de Développement

### 1. Manipulation de la Donnée
- Chargement depuis les dossiers
- Normalisation [0, 1]
- Encodage des labels
- Augmentation de données avec ImageDataGenerator
- Visualisation des transformations

### 2. Entraînement de 3 Modèles
- Implémentation de 2 callbacks (EarlyStopping, ReduceLROnPlateau)
- CNN from scratch avec Sequential et Dropout
- Fine-tuning de VGG16 pré-entraîné sur ImageNet
- Fine-tuning de ResNet50 pré-entraîné sur ImageNet
- Sauvegarde des poids des 3 modèles

### 3. Test des Modèles
- Calcul de la matrice de confusion
- Calcul de toutes les métriques (accuracy, precision, recall, f1-score, sensibilité, spécificité)
- Affichage des courbes ROC et calcul de l'AUC
- Interprétation et comparaison des résultats

## 📚 Références

- Dataset: NIH Malaria Datasets - https://ceb.nlm.nih.gov/repositories/malaria-datasets/
- VGG16: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- ResNet50: "Deep Residual Learning for Image Recognition"
- Transfer Learning: Utilisation de modèles pré-entraînés sur ImageNet

## 👥 Auteurs

Projet réalisé dans le cadre du FOAD du 05/02/2026

## 📄 Licence

Ce projet est fourni à des fins éducatives.
