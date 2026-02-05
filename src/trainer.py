from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from src.config import *

class Trainer:
    
    def __init__(self, model, model_name):
        self.model = model.model
        self.model_name = model_name
        
    def train(self, X_train, y_train, X_val, y_val, augmentation=None):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        if augmentation:
            history = self.model.fit(
                augmentation.flow(X_train, y_train, batch_size=TAILLE_BATCH),
                validation_data=(X_val, y_val),
                epochs=NOMBRE_EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=TAILLE_BATCH,
                epochs=NOMBRE_EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
        
        self.model.save_weights(os.path.join(CHEMIN_RESULTATS, f'{self.model_name}.weights.h5'))
        
        return history
