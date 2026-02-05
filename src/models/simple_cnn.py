from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization

class SimpleCNN:
    
    def __init__(self, image_size=64):
        self.image_size = image_size
        self.model = self._build()
        
    def _build(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size, self.image_size, 3)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
