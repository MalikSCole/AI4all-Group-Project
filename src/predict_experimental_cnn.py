import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from train_sam_cnn import SamResNet18MultiTask, ordinal_logits_to_class

def predict_food_nutrition(image_path, model_path):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(image_path):
        print(f"[-] Error: Image file not found at: {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"[-] Error: Trained model checkpoint not found at: {model_path}")
        print("Please run 'src/train_sam_cnn.py' first to train the model.")
        return
        
    print(f"[+] Loading trained model from {model_path}...")
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model (backbone can remain frozen since we only do inference)
    model = SamResNet18MultiTask(pretrained=False, freeze_backbone=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load StandardScaler
    scaler = checkpoint['scaler']
    class_names = checkpoint['class_names']
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"[+] Loading and preprocessing image: {image_path}...")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    
    # Inference
    print("[+] Running model inference...")
    with torch.no_grad():
        ordinal_outputs, regression_outputs = model(image_tensor)
        
        # Predict class
        predicted_class_idx = ordinal_logits_to_class(ordinal_outputs).item()
        predicted_class = class_names[predicted_class_idx]
        
        # Unscale regression values
        scaled_regression_np = regression_outputs.cpu().numpy()
        unscaled_regression = scaler.inverse_transform(scaled_regression_np)[0]
        
    calories = unscaled_regression[0]
    mass = unscaled_regression[1]
    fat = unscaled_regression[2]
    carb = unscaled_regression[3]
    protein = unscaled_regression[4]
    
    print("\n" + "="*50)
    print("             PREDICTION RESULTS")
    print("="*50)
    print(f" * Calorie Category   : {predicted_class}")
    print(f" * Estimated Calories : {calories:.2f} kcal")
    print(f" * Estimated Mass     : {mass:.2f} grams")
    print(f" * Estimated Fat      : {fat:.2f} grams")
    print(f" * Estimated Carbs    : {carb:.2f} grams")
    print(f" * Estimated Protein  : {protein:.2f} grams")
    print("="*50 + "\n")

def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_dir, 'models', 'sam_cnn_model.pth')
    
    # Default image if none provided
    default_image = os.path.join(project_dir, 'AI4All Dataset', 'imagery', 'realsense_overhead', 'dish_1556572657', 'rgb.png')
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else default_image
    predict_food_nutrition(image_path, model_path)

if __name__ == '__main__':
    main()
