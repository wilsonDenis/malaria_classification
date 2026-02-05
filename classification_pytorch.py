import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("GPU Apple Silicon (MPS) disponible")
else:
    device = torch.device('cpu')
    print("Utilisation du CPU")

plt.style.use('seaborn-v0_8-darkgrid')

taille_image = 64
taille_batch = 32
nombre_epochs = 30
taux_apprentissage = 0.001
patience_early_stopping = 5

classes = ['Parasitee', 'Non_parasitee']
nombre_classes = 2

chemin_donnees = './data/malaria_hematie_dataset'
chemin_resultats = './resultats'
os.makedirs(chemin_resultats, exist_ok=True)

print(f"\nConfiguration:")
print(f"  - Taille image: {taille_image}x{taille_image}")
print(f"  - Batch size: {taille_batch}")
print(f"  - Epochs max: {nombre_epochs}")
print(f"  - Learning rate: {taux_apprentissage}")
print(f"  - Device: {device}")


class MalariaDataset(Dataset):
    
    def __init__(self, chemins_images, labels, transform=None):
        self.chemins_images = chemins_images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.chemins_images)
    
    def __getitem__(self, index):
        chemin_image = self.chemins_images[index]
        image = Image.open(chemin_image).convert('RGB')
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


print("\n" + "="*60)
print("CHARGEMENT DES DONNÉES")
print("="*60)

chemins_images_liste = []
labels_liste = []

for index_classe, nom_classe in enumerate(classes):
    chemin_classe = os.path.join(chemin_donnees, nom_classe)
    
    if not os.path.exists(chemin_classe):
        print(f"Attention: {chemin_classe} n'existe pas")
        continue
    
    for nom_image in os.listdir(chemin_classe):
        chemin_complet = os.path.join(chemin_classe, nom_image)
        chemins_images_liste.append(chemin_complet)
        labels_liste.append(index_classe)

print(f"\nTotal d'images chargées: {len(chemins_images_liste)}")
for index_classe, nom_classe in enumerate(classes):
    nombre_images_classe = labels_liste.count(index_classe)
    print(f"  - {nom_classe}: {nombre_images_classe} images")


chemins_train, chemins_temp, labels_train, labels_temp = train_test_split(
    chemins_images_liste, labels_liste, test_size=0.3, random_state=42, stratify=labels_liste
)

chemins_val, chemins_test, labels_val, labels_test = train_test_split(
    chemins_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp
)

print(f"\nRépartition des données:")
print(f"  - Train: {len(chemins_train)} images")
print(f"  - Validation: {len(chemins_val)} images")
print(f"  - Test: {len(chemins_test)} images")


transform_train = transforms.Compose([
    transforms.Resize((taille_image, taille_image)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((taille_image, taille_image)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = MalariaDataset(chemins_train, labels_train, transform=transform_train)
dataset_val = MalariaDataset(chemins_val, labels_val, transform=transform_test)
dataset_test = MalariaDataset(chemins_test, labels_test, transform=transform_test)

dataloader_train = DataLoader(dataset_train, batch_size=taille_batch, shuffle=True, num_workers=2)
dataloader_val = DataLoader(dataset_val, batch_size=taille_batch, shuffle=False, num_workers=2)
dataloader_test = DataLoader(dataset_test, batch_size=taille_batch, shuffle=False, num_workers=2)

print("\nDataLoaders créés avec succès")


print("\n" + "="*60)
print("VISUALISATION DES ÉCHANTILLONS")
print("="*60)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for index_image in range(10):
    row = index_image // 5
    col = index_image % 5
    
    image = Image.open(chemins_train[index_image])
    label_index = labels_train[index_image]
    
    axes[row, col].imshow(image)
    axes[row, col].set_title(f"{classes[label_index]}", fontsize=12, fontweight='bold')
    axes[row, col].axis('off')

plt.suptitle("Exemples d'Images du Dataset", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{chemin_resultats}/exemples_images.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Visualisation sauvegardée: {chemin_resultats}/exemples_images.png")


class CNNSimple(nn.Module):
    
    def __init__(self, nombre_classes=2):
        super(CNNSimple, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, nombre_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def creer_vgg16(nombre_classes=2):
    modele = models.vgg16(weights='DEFAULT')
    
    for param in modele.features.parameters():
        param.requires_grad = False
    
    for param in modele.features[-6:].parameters():
        param.requires_grad = True
    
    modele.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 2 * 2, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, nombre_classes)
    )
    
    return modele


def creer_resnet50(nombre_classes=2):
    modele = models.resnet50(weights='DEFAULT')
    
    for param in modele.parameters():
        param.requires_grad = False
    
    for param in modele.layer4.parameters():
        param.requires_grad = True
    
    nombre_features = modele.fc.in_features
    modele.fc = nn.Sequential(
        nn.Linear(nombre_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, nombre_classes)
    )
    
    return modele


def entrainer_modele(modele, nom_modele, dataloader_train, dataloader_val, nombre_epochs, patience):
    
    modele = modele.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, modele.parameters()), lr=taux_apprentissage)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    meilleure_val_loss = float('inf')
    compteur_patience = 0
    
    historique_train_loss = []
    historique_val_loss = []
    historique_train_acc = []
    historique_val_acc = []
    
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT: {nom_modele}")
    print(f"{'='*60}")
    
    for epoch in range(nombre_epochs):
        
        modele.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        
        barre_progression = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{nombre_epochs}')
        
        for batch_images, batch_labels in barre_progression:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = modele(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_images.size(0)
            _, predictions = torch.max(outputs, 1)
            total_train_correct += (predictions == batch_labels).sum().item()
            total_train_samples += batch_images.size(0)
            
            barre_progression.set_postfix({'loss': loss.item()})
        
        
        modele.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch_images, batch_labels in dataloader_val:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = modele(batch_images)
                loss = criterion(outputs, batch_labels)
                
                total_val_loss += loss.item() * batch_images.size(0)
                _, predictions = torch.max(outputs, 1)
                total_val_correct += (predictions == batch_labels).sum().item()
                total_val_samples += batch_images.size(0)
        
        epoch_train_loss = total_train_loss / total_train_samples
        epoch_train_acc = total_train_correct / total_train_samples
        epoch_val_loss = total_val_loss / total_val_samples
        epoch_val_acc = total_val_correct / total_val_samples
        
        historique_train_loss.append(epoch_train_loss)
        historique_val_loss.append(epoch_val_loss)
        historique_train_acc.append(epoch_train_acc)
        historique_val_acc.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < meilleure_val_loss:
            meilleure_val_loss = epoch_val_loss
            compteur_patience = 0
            torch.save(modele.state_dict(), f'{chemin_resultats}/{nom_modele}_meilleur.pth')
            print(f'✓ Meilleur modèle sauvegardé')
        else:
            compteur_patience += 1
            if compteur_patience >= patience:
                print(f'\nEarly stopping à l\'epoch {epoch+1}')
                break
    
    modele.load_state_dict(torch.load(f'{chemin_resultats}/{nom_modele}_meilleur.pth'))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(historique_train_loss, label='Train Loss', marker='o')
    ax1.plot(historique_val_loss, label='Val Loss', marker='s')
    ax1.set_title(f'{nom_modele} - Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(historique_train_acc, label='Train Accuracy', marker='o')
    ax2.plot(historique_val_acc, label='Val Accuracy', marker='s')
    ax2.set_title(f'{nom_modele} - Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{chemin_resultats}/{nom_modele}_courbes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return modele, {
        'train_loss': historique_train_loss,
        'val_loss': historique_val_loss,
        'train_acc': historique_train_acc,
        'val_acc': historique_val_acc
    }


def evaluer_modele(modele, nom_modele, dataloader_test):
    
    modele.eval()
    
    toutes_predictions = []
    tous_labels = []
    toutes_probas = []
    
    with torch.no_grad():
        for batch_images, batch_labels in tqdm(dataloader_test, desc='Évaluation'):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = modele(batch_images)
            probas = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            toutes_predictions.extend(predictions.cpu().numpy())
            tous_labels.extend(batch_labels.cpu().numpy())
            toutes_probas.extend(probas.cpu().numpy())
    
    toutes_predictions = np.array(toutes_predictions)
    tous_labels = np.array(tous_labels)
    toutes_probas = np.array(toutes_probas)
    
    matrice_confusion = confusion_matrix(tous_labels, toutes_predictions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(matrice_confusion, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title(f'Matrice de Confusion - {nom_modele}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Vraie classe')
    ax1.set_xlabel('Classe prédite')
    
    fpr, tpr, _ = roc_curve(tous_labels, toutes_probas[:, 1])
    auc_score = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Faux Positifs')
    ax2.set_ylabel('Vrais Positifs')
    ax2.set_title(f'Courbe ROC - {nom_modele}', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{chemin_resultats}/{nom_modele}_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS - {nom_modele}")
    print(f"{'='*60}")
    print(classification_report(tous_labels, toutes_predictions, target_names=classes))
    print(f"AUC: {auc_score:.4f}")
    
    return classification_report(tous_labels, toutes_predictions, target_names=classes, output_dict=True)


print("\n" + "="*60)
print("ENTRAÎNEMENT DES MODÈLES")
print("="*60)

modele_cnn, historique_cnn = entrainer_modele(
    CNNSimple(nombre_classes), 'CNN_Simple', 
    dataloader_train, dataloader_val, nombre_epochs, patience_early_stopping
)

modele_vgg, historique_vgg = entrainer_modele(
    creer_vgg16(nombre_classes), 'VGG16',
    dataloader_train, dataloader_val, nombre_epochs, patience_early_stopping
)

modele_resnet, historique_resnet = entrainer_modele(
    creer_resnet50(nombre_classes), 'ResNet50',
    dataloader_train, dataloader_val, nombre_epochs, patience_early_stopping
)


print("\n" + "="*60)
print("ÉVALUATION SUR LE TEST SET")
print("="*60)

resultats_cnn = evaluer_modele(modele_cnn, 'CNN_Simple', dataloader_test)
resultats_vgg = evaluer_modele(modele_vgg, 'VGG16', dataloader_test)
resultats_resnet = evaluer_modele(modele_resnet, 'ResNet50', dataloader_test)


print("\n" + "="*60)
print("COMPARAISON DES MODÈLES")
print("="*60)

metriques = ['precision', 'recall', 'f1-score']
modeles_noms = ['CNN Simple', 'VGG16', 'ResNet50']
resultats_liste = [resultats_cnn, resultats_vgg, resultats_resnet]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for index_metrique, nom_metrique in enumerate(metriques):
    valeurs = [resultats['weighted avg'][nom_metrique] for resultats in resultats_liste]
    
    axes[index_metrique].bar(modeles_noms, valeurs, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[index_metrique].set_title(nom_metrique.capitalize(), fontsize=14, fontweight='bold')
    axes[index_metrique].set_ylim([0, 1])
    axes[index_metrique].grid(True, alpha=0.3, axis='y')
    
    for index_valeur, valeur in enumerate(valeurs):
        axes[index_metrique].text(index_valeur, valeur + 0.02, f'{valeur:.3f}', 
                                   ha='center', fontweight='bold')

plt.suptitle('Comparaison des Performances', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{chemin_resultats}/comparaison_modeles.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Tous les résultats sont sauvegardés dans:", chemin_resultats)
print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ!")
print("="*60)
