import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import CNNModel
from config.config import transform, NUM_CLASSES

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def predict_image(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def visualize_prediction(image_path, predicted_class, confidence):
    class_names = {
        0: 'İnme Yok/Kronik/Diğer',
        1: 'İskemi',
        2: 'Kanama'
    }
    
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction results
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.6, f'Predicted: {class_names[predicted_class]}', 
             horizontalalignment='center', fontsize=12)
    plt.text(0.5, 0.4, f'Confidence: {confidence:.2%}', 
             horizontalalignment='center', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    model_path = 'outputs/best_model.pth'
    model, device = load_model(model_path)
    
    # Test bir görüntü seçin
    data_dir = './data'
    test_image = None
    
    # İlk bulunan görüntüyü al
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(class_path, img_name)
                    break
        if test_image:
            break
    
    if test_image:
        predicted_class, confidence = predict_image(model, test_image, device)
        visualize_prediction(test_image, predicted_class, confidence)
    else:
        print("No test image found in data directory")

if __name__ == '__main__':
    main() 