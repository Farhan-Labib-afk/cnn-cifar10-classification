# CIFAR-10 CNN Classification

Train a custom CNN and a ResNet18 transfer model on CIFAR-10, compare results, and visualize performance.

## Project Structure

- `cnn_classifier.py`: training + evaluation
- `app.py`: Streamlit results dashboard
- `results.json`: final experiment summary
- `assets/`: plots and visual outputs
- `models/`: saved model weights
- `data/`: CIFAR-10 dataset (auto-downloaded)

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
.env\Scripts\python.exe -m pip install -r requirements.txt
```

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```powershell
.env\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you do NOT have an NVIDIA GPU (CPU-only):

```powershell
.env\Scripts\python.exe -m pip install torch torchvision torchaudio
```

### macOS

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install PyTorch (CPU/MPS build from PyPI):

```bash
python3 -m pip install torch torchvision torchaudio
```

## Train + Evaluate

```powershell
.env\Scripts\python.exe cnn_classifier.py
```

Outputs:
- Models saved in `models/`
- Plots saved in `assets/`
- Metrics printed to console

## Results Dashboard

```powershell
.env\Scripts\python.exe -m streamlit run app.py
```

Use the sidebar to save `results.json` and refresh assets.

## Latest Run (Best Scores)

- Custom CNN best accuracy: 86.76%
- ResNet18 best accuracy: 84.15%
- Custom CNN training time: 11.94 minutes
- ResNet18 training time: 7.66 minutes

## Screenshots

### Training Curves

![Training Curves](assets/training_comparison.png)

### Custom CNN Confusion Matrix

![Custom CNN Confusion Matrix](assets/custom_cnn_confusion_matrix.png)

### ResNet18 Confusion Matrix

![ResNet18 Confusion Matrix](assets/resnet18_confusion_matrix.png)

### Sample Predictions

![Sample Predictions](assets/sample_predictions.png)

### Streamlit Dashboard

![Streamlit Dashboard](assets/streamlit_dashboard.png)

## Author

Farhan Labib
