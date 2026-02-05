import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from src.config import *

class Evaluator:
    
    def __init__(self, model, model_name):
        self.model = model.model
        self.model_name = model_name
        
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de Confusion - {self.model_name}')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.savefig(os.path.join(CHEMIN_RESULTATS, f'{self.model_name}_confusion.png'))
        plt.close()
        
        print(f"\n{'='*50}")
        print(f"RÉSULTATS - {self.model_name}")
        print(f"{'='*50}")
        print(classification_report(y_true, y_pred, target_names=CLASSES))
        
        return classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
