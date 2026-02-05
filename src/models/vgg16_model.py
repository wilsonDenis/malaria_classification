from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

class VGG16Model:
    
    def __init__(self, image_size=64):
        self.image_size = image_size
        self.model = self._build()
    
    def _build(self):
        base = VGG16(weights='imagenet', include_top=False, 
                     input_shape=(self.image_size, self.image_size, 3))
        
        for layer in base.layers[:-4]:
            layer.trainable = False
        
        x = Flatten()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=base.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
