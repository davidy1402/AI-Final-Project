import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Import our custom modules
from data_loader import get_dataloaders
from model import get_flood_model

# =====================================================================
# CONFIGURATION CONSTANTS
# =====================================================================
# If training on a CPU or memory-constrained machine, you should reduce:
# BATCH_SIZE = 16
# NUM_EPOCHS = 8
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# =====================================================================

def train_model():
    """
    Main training and evaluation pipeline for the Flood Damage Assessment System.
    Trains the ResNet-50 model on the training subset, validates on validation,
    and runs final classification reporting and confusion matrix generation
    on the test subset using the best weights.
    """
    # 1. Device detection (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using hardware device: {device}")
    
    # Create save directories if they do not exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 2. Get DataLoaders
    print("Initializing DataLoaders...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure that you have run 'python generate_mock_data.py' or have downloaded the dataset.")
        return
        
    # 3. Create Model, Loss, Optimizer
    model = get_flood_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    # Filter to only optimize parameters that require grad (layer3, layer4, and fc)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Variables to track training progress
    best_val_acc = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # 4. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
              
        # Save best model checkpoint based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  => Saved best model checkpoint to {best_model_path} with Val Acc: {best_val_acc:.4f}")
            
    print(f"\nTraining completed! Best Validation Accuracy achieved: {best_val_acc:.4f}")
    
    # 5. Final Evaluation on Test Split
    print("\n--- Running Evaluation on Test Split ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best weights for final test evaluation.")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Generate and save metrics
    target_names = ["Not Flooded", "Flooded"]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print("\nTest Split Classification Report:\n", report)
    
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("=== Drone Aerial Flood Damage Assessment - Test Split Evaluation ===\n")
        f.write(f"Best Validation Accuracy during training: {best_val_acc:.4f}\n\n")
        f.write(report)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate and save confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix (Test Split)")
    
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix image saved to {cm_path}")

if __name__ == "__main__":
    train_model()
