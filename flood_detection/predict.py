import os
import argparse
import torch
from PIL import Image

# Import our custom modules
from data_loader import get_inference_transform
from model import get_flood_model

def predict(image_path, model_path=None):
    """
    Predicts whether a single input image shows a Flooded or Not Flooded area
    using the trained ResNet-50 binary classifier.

    Args:
        image_path (str): Absolute or relative path to the image.
        model_path (str, optional): Path to the saved model checkpoint (.pth). 
                                   Defaults to checkpoints/best_model.pth.

    Returns:
        tuple: (predicted_label: str, confidence_score: float)
    """
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Please run 'python train.py' first to train the model and save the checkpoint."
        )
        
    # 1. Device detection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # 2. Load model and map to device
    # Set pretrained=False since we load custom weights immediately
    model = get_flood_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. Load image and apply validation transform
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image not found at '{image_path}'.")
        
    img = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension: [1, 3, 224, 224]
    
    # 4. Model forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    class_idx = predicted_class.item()
    confidence_pct = confidence.item() * 100
    
    # Class mapping
    labels_map = {0: "Not Flooded", 1: "Flooded"}
    predicted_label = labels_map[class_idx]
    
    return predicted_label, confidence_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference CLI for Drone Aerial Flood Damage Assessment")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth checkpoint")
    args = parser.parse_args()
    
    try:
        label, confidence = predict(args.image, args.checkpoint)
        print("\n" + "="*30)
        print("    CLASSIFICATION SUCCESSFUL")
        print("="*30)
        print(f"Image File: {os.path.basename(args.image)}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*30 + "\n")
    except Exception as e:
        print(f"Error during prediction: {e}")
