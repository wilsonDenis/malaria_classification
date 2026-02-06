"""
Script principal pour la classification de malaria.
Entraîne et évalue les modèles CNN sur le dataset de cellules sanguines.
"""

from src.data_manager import DataManager
from src.models.simple_cnn import SimpleCNN
from src.models.vgg16_model import VGG16Model
from src.models.resnet50_model import ResNet50Model
from src.trainer import Trainer
from src.evaluator import Evaluator


def main():
    # Étape 1: Chargement des données
    print("\n[1/5] Chargement des données...")
    gestionnaire_donnees = DataManager()
    gestionnaire_donnees.load_data()
    
    # Création des DataLoaders
    chargeur_train, chargeur_val, chargeur_test = gestionnaire_donnees.get_dataloaders(avec_augmentation=True)
    
    print(f"✓ Train: {len(chargeur_train.dataset)}, Val: {len(chargeur_val.dataset)}, Test: {len(chargeur_test.dataset)}")
    
    # Étape 2: Entraînement CNN Simple
    print("\n[2/5] Entraînement CNN Simple...")
    modele_cnn = SimpleCNN()
    entraineur_cnn = Trainer(modele_cnn, 'CNN_Simple')
    historique_cnn = entraineur_cnn.train(chargeur_train, chargeur_val)
    
    # Étape 3: Entraînement VGG16
    print("\n[3/5] Entraînement VGG16...")
    modele_vgg = VGG16Model()
    entraineur_vgg = Trainer(modele_vgg, 'VGG16')
    historique_vgg = entraineur_vgg.train(chargeur_train, chargeur_val)
    
    # Étape 4: Entraînement ResNet50
    print("\n[4/5] Entraînement ResNet50...")
    modele_resnet = ResNet50Model()
    entraineur_resnet = Trainer(modele_resnet, 'ResNet50')
    historique_resnet = entraineur_resnet.train(chargeur_train, chargeur_val)
    
    # Étape 5: Évaluation des modèles
    print("\n[5/5] Évaluation des modèles...")
    
    evaluateur_cnn = Evaluator(modele_cnn, 'CNN_Simple')
    rapport_cnn = evaluateur_cnn.evaluate(chargeur_test)
    
    evaluateur_vgg = Evaluator(modele_vgg, 'VGG16')
    rapport_vgg = evaluateur_vgg.evaluate(chargeur_test)
    
    evaluateur_resnet = Evaluator(modele_resnet, 'ResNet50')
    rapport_resnet = evaluateur_resnet.evaluate(chargeur_test)
    
    # Résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    print(f"   CNN Simple:  {rapport_cnn['accuracy']*100:.2f}%")
    print(f"   VGG16:       {rapport_vgg['accuracy']*100:.2f}%")
    print(f"   ResNet50:    {rapport_resnet['accuracy']*100:.2f}%")
    print("="*60)
    print("✅ Résultats sauvegardés dans ./resultats/")


if __name__ == "__main__":
    main()
