from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

class ResNet50Model:
    
    def __init__(self, image_size=64):
        self.image_size = image_size
        self.model = self._build()
    
    def _build(self):
        base = ResNet50(weights='imagenet', include_top=False,
                        input_shape=(self.image_size, self.image_size, 3))
        
        for layer in base.layers[:-10]:
            layer.trainable = False
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=base.input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
