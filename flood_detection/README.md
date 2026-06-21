# Drone Aerial Flood Damage Assessment System

This repository contains a deep learning application designed to assess flood damage from drone/UAV aerial images, specifically developed as a group final project for **SOF106 Principles of Artificial Intelligence**. The system utilizes a pre-trained ResNet-50 network fine-tuned using PyTorch for binary image classification (`Flooded` vs. `Not Flooded`). The application is designed to analyze disaster scenes and generate triage maps, featuring a custom command-line prediction interface and an interactive human-in-the-loop Streamlit web application.

## Folder Structure

```
flood_detection/
├── data/               # FloodNet images go here
├── results/            # Confusion matrix, feedback log, and metrics saved here
├── checkpoints/        # Best model checkpoint (.pth) saved here
├── data_loader.py      # Data preparation, splitting, and augmentation
├── model.py            # Pretrained ResNet-50 transfer learning model
├── train.py            # Model training and testing script
├── predict.py          # Command line prediction interface
├── app.py              # Interactive Streamlit Web application
├── generate_mock_data.py # Helper script to create synthetic verification images
└── requirements.txt    # Project dependencies
```

## Setup Instructions

First, navigate to the project directory and install the necessary package dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*Note: Make sure PyTorch and Torchvision are installed. If you are using a GPU or Apple Silicon (MPS), PyTorch will automatically utilize the hardware acceleration during training and inference.*

## How to Run

Follow these steps in sequence to test and run the project:

### 1. Generate Mock Data (For Testing)
If you do not have the full FloodNet dataset downloaded yet, you can create a synthetic dataset of 64 images under the correct directory tree structure to verify the training pipeline:

```bash
python generate_mock_data.py
```

### 2. Train the Model
Train the ResNet-50 binary classifier on the dataset. The script automatically splits the data, runs training/validation, saves the best weights, and writes validation reports and a confusion matrix:

```bash
python train.py
```

*Tip: If you are running on a CPU-only computer, you can open `train.py` and modify `BATCH_SIZE = 16` and `NUM_EPOCHS = 8` to reduce training time.*

### 3. Run Single Image Inference (CLI)
Run a classification prediction on any single input image path using the CLI interface:

```bash
python predict.py --image data/Train/Labeled/Flooded/image/mock_flooded_0.jpg
```

### 4. Run the Streamlit Web Application
Launch the interactive web application to upload drone images, view prediction results color-coded dynamically (red for Flooded, green for Not Flooded), see confidence progress bars, and log correction feedback to `results/feedback_log.csv`:

```bash
streamlit run app.py
```
