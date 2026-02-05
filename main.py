from src.data_manager import DataManager
from src.models import SimpleCNN, VGG16Model, ResNet50Model
from src.trainer import Trainer
from src.evaluator import Evaluator

def main():
    print("="*60)
    print("CLASSIFICATION DU PALUDISME")
    print("="*60)
    
    print("\n[1/4] Chargement des données...")
    data = DataManager().load_data()
    print(f"Train: {len(data.X_train)}, Val: {len(data.X_val)}, Test: {len(data.X_test)}")
    
    augmentation = data.get_augmentation()
    
    print("\n[2/4] Entraînement CNN Simple...")
    cnn = SimpleCNN()
    Trainer(cnn, 'CNN_Simple').train(data.X_train, data.y_train, data.X_val, data.y_val, augmentation)
    
    print("\n[3/4] Entraînement VGG16...")
    vgg = VGG16Model()
    Trainer(vgg, 'VGG16').train(data.X_train, data.y_train, data.X_val, data.y_val)
    
    print("\n[4/4] Entraînement ResNet50...")
    resnet = ResNet50Model()
    Trainer(resnet, 'ResNet50').train(data.X_train, data.y_train, data.X_val, data.y_val)
    
    print("\nÉvaluation...")
    Evaluator(cnn, 'CNN_Simple').evaluate(data.X_test, data.y_test)
    Evaluator(vgg, 'VGG16').evaluate(data.X_test, data.y_test)
    Evaluator(resnet, 'ResNet50').evaluate(data.X_test, data.y_test)
    
    print("\n" + "="*60)
    print("TERMINÉ!")
    print("="*60)

if __name__ == "__main__":
    main()
