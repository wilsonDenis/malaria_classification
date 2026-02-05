import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from src.config import *

class DataManager:
    
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self):
        images = []
        labels = []
        
        for classe in CLASSES:
            chemin = os.path.join(CHEMIN_DONNEES, classe)
            if not os.path.exists(chemin):
                continue
                
            for img_name in os.listdir(chemin):
                try:
                    img = Image.open(os.path.join(chemin, img_name)).convert('RGB')
                    img = img.resize((TAILLE_IMAGE, TAILLE_IMAGE))
                    images.append(np.array(img) / 255.0)
                    labels.append(classe)
                except:
                    continue
        
        images = np.array(images)
        labels = LabelEncoder().fit_transform(labels)
        labels = to_categorical(labels, 2)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.3, random_state=42
        )
        
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        self.X_train = X_train
        self.y_train = y_train
        
        return self
    
    def get_augmentation(self):
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
