# CIFAR-10 CNN Classification

Train a custom CNN and a ResNet18 transfer model on CIFAR-10, compare results, and visualize performance.

## Project Structure

## Dataset

CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

- `cnn_classifier.py`: training + evaluation
- `app.py`: Streamlit results dashboard
- `results.json`: final experiment summary
- `assets/`: plots and visual outputs
- `models/`: saved model weights
- `data/`: CIFAR-10 dataset (auto-downloaded)

## Key Takeaways

- A custom-designed CNN (3.25M parameters) outperformed a pretrained ResNet18 (11.18M parameters) on CIFAR-10.
- Demonstrates that task-specific architecture design can outperform larger pretrained models.
- Highlights tradeoffs between model size, accuracy, and training time.

## Models

### Custom CNN (From Scratch)
- 3 convolutional blocks with BatchNorm, ReLU, and MaxPooling
- Dropout-based regularization in the classifier
- Designed specifically for 32×32 CIFAR-10 images
- ~3.25M parameters

### ResNet18 (Transfer Learning)
- Pretrained on ImageNet
- Final classification layer replaced for CIFAR-10
- Fine-tuned using Adam optimizer
- ~11.18M parameters

## Skills Demonstrated

- Convolutional Neural Network (CNN) design
- Transfer learning with pretrained models
- GPU-accelerated training using PyTorch
- Data augmentation and regularization techniques
- Model evaluation using confusion matrices and classification reports
- Experiment tracking and result visualization

## Quick Start

```bash
git clone https://github.com/<your-username>/cnn-cifar10-classification.git
cd cnn-cifar10-classification
python -m venv .venv
```

## Setup

A virtual environment is recommended to keep dependencies isolated and avoid conflicts. You can also run this on your system Python if you prefer, but you may run into version clashes with other projects.

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

NVIDIA GPU (CUDA):

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU-only:

```powershell
python -m pip install torch torchvision torchaudio
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install torch torchvision torchaudio
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

NVIDIA GPU (CUDA):

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU-only:

```bash
python3 -m pip install torch torchvision torchaudio
```

## Train + Evaluate

Windows (PowerShell):

```powershell
.\venv\Scripts\python.exe cnn_classifier.py
```

macOS/Linux:

```bash
python3 cnn_classifier.py
```

Outputs:
- Models saved in `models/`
- Plots saved in `assets/`
- Metrics printed to console

## Results Dashboard

Windows (PowerShell):

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```bash
python3 -m streamlit run app.py
```

Use the sidebar to save `results.json`.

## Latest Run (Best Scores)

- Custom CNN best accuracy: 86.76%
- ResNet18 best accuracy: 84.15%
- Custom CNN training time: 11.94 minutes
- ResNet18 training time: 7.66 minutes

## Screenshots

### Streamlit Dashboard

![Streamlit Dashboard](assets/streamlit_dashboard.png)

### Training Curves

![Training Curves](assets/training_comparison.png)

### Custom CNN Confusion Matrix

![Custom CNN Confusion Matrix](assets/custom_cnn_confusion_matrix.png)

### ResNet18 Confusion Matrix

![ResNet18 Confusion Matrix](assets/resnet18_confusion_matrix.png)

### Sample Predictions

![Sample Predictions](assets/sample_predictions.png)

## Author

Farhan Labib
